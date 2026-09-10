"""Rebuilds <season>_map_rectified.png (+ its .json extent sidecar) from a
raw game screenshot.

Run this whenever you have a new/better source screenshot for a season. It
does NOT run at app startup - hex_pathfinding_demo.py just loads the files
this script produces (named in landInfo.json's "map_files").

What it does:

0. Self-calibrate a rough (ir, ic) <-> pixel affine estimate for THIS
   specific source image (its resolution/crop can change any time the user
   re-exports a screenshot, so nothing is hardcoded): locate the unique 'ST'
   start hex by its distinctive highlight-border color as one anchor point,
   then estimate pixel-per-hex spacing from the bounding box of all detected
   hex blobs versus the known (ir, ic) extent of real terrain in
   map_<season>.csv. This only needs to be roughly right - it's just the seed for
   nearest-integer (ir, ic) assignment in step 1.

1. Detect every hex's fill-color blob via connected-component analysis, and
   assign each blob an (ir, ic) grid coordinate using the rough affine
   estimate, refined by a few passes of least-squares fitting + outlier
   rejection. This gives ~1000+ (image pixel <-> internal hex) correspondence
   points and, from the final fit pass, one precise global affine transform
   (column step, row step, half-row offset for odd columns).

2. Place the WHOLE original screenshot - untouched, uncropped, no
   transparency - using that single rigid affine transform (downsampled only
   if needed to stay under OUTPUT_LONG_EDGE, for redraw performance). No
   per-hex warping and no "is this pixel real terrain" masking: an earlier
   version of this script did a per-hex scipy.interpolate.griddata warp (see
   git history) to correct for a source screenshot that turned out to be a
   stitched composite with per-row drift a single affine couldn't capture,
   but that approach's real-terrain mask was clipping the outer rim off
   edge-of-island hexes, and per user feedback a fully intact image is
   preferred over pixel-perfect edges anyway. Only reach for the per-hex
   warp again if a future source image's fit residual (the "final xerr/yerr
   std" printed during the run) stays large (tens of px) instead of
   converging to near-zero - that's the signature of a stitched image a
   single affine can't align end-to-end.

Usage:
    cd tools
    python rectify_map_image.py --season S25 --src ../S25_map.png \
        --out ../S25_map_rectified_transparent.png --st-pixel 6480 6885

Requires: numpy, scipy, Pillow (all already project dependencies).
Inputs (in the project root, one level up): the source screenshot (RGBA
preferred, e.g. from tools/remove_map_background.py - it does not need a
perfect transparent background, alpha is only used to help separate real
content from background during blob detection; an RGB JPG also works if its
background is darker than the hexes), plus map_<season>.csv (the unique 'ST'
cell marks the start hex) and landInfo.json.
Outputs: ../<season>_map_rectified.png (or --out) and
../<season>_map_rectified_extent.json (the imshow() placement extent
hex_pathfinding_demo.py reads at runtime).

If step 0's automatic ST-anchor detection fails (prints a warning / 0 pixels
found), the source image's UI no longer highlights the current team position
in the same orange/green style - open the image, find the "1队" (or similar)
highlighted hex near the team's start position, and pass --st-pixel X Y
(a pixel roughly in the middle of that hex).
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--season', default='S25',
                   help='season prefix: reads map_<season>.csv, writes <season>_map_rectified*.png/.json')
    p.add_argument('--src', help='source screenshot (default: <season>_map.png, else <season>_map.jpg)')
    p.add_argument('--out', help='output PNG (default: <season>_map_rectified.png)')
    p.add_argument('--st-pixel', nargs=2, type=float, metavar=('X', 'Y'),
                   help="pixel near the centre of the highlighted 'current team position' hex; "
                        "bypasses auto-detection")
    return p.parse_args()


_ARGS = _parse_args()
SEASON = _ARGS.season
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_IMAGE_PNG = os.path.join(PROJECT_ROOT, f'{SEASON}_map.png')
SRC_IMAGE_JPG = os.path.join(PROJECT_ROOT, f'{SEASON}_map.jpg')
MAP_CSV = os.path.join(PROJECT_ROOT, f'map_{SEASON}.csv')
LAND_INFO = os.path.join(PROJECT_ROOT, 'landInfo.json')
OUTPUT_PNG = _ARGS.out or os.path.join(PROJECT_ROOT, f'{SEASON}_map_rectified.png')
EXTENT_JSON = os.path.join(PROJECT_ROOT, f'{SEASON}_map_rectified_extent.json')

HEX_SIZE = 5.0
X_SCALE = 1.0
Y_SCALE = 0.6
ROWS = COLS = 60
OUTPUT_LONG_EDGE = 2600  # keep in sync with app's own display budget

# Color range of the unique 'ST' start hex's distinctive in-game highlight
# border (orange/red ring around the team's current-position marker). Used
# only to seed the rough affine estimate.
ST_COLOR_RANGE = dict(r_lo=140, r_hi=220, g_lo=40, g_hi=130, b_hi=80)

# Auto-detection of the ST highlight color is unreliable when similar tan/
# orange terrain tones appear elsewhere in the image (it did on S24_map.png -
# matched a terrain patch instead of the actual highlight), so --st-pixel
# bypasses it entirely.
ST_PIXEL_OVERRIDE = tuple(_ARGS.st_pixel) if _ARGS.st_pixel else None


def _center(ir, ic):
    x = ic * 1.5 * HEX_SIZE
    y = ir * np.sqrt(3) * HEX_SIZE
    if ic % 2 == 1:
        y += np.sqrt(3) / 2 * HEX_SIZE
    return x, y


def _load_terrain_extent():
    with open(MAP_CSV, newline='') as f:
        raw = [row for row in csv.reader(f)]
    raw.reverse()
    with open(LAND_INFO, encoding='utf-8') as f:
        land_info = json.load(f)
    non_empty = []
    for ir in range(ROWS):
        for ic in range(COLS):
            name = land_info.get(raw[ir][ic], land_info.get('default', {})).get('name', 'empty')
            if name != 'empty':
                non_empty.append((ir, ic))
    irs = [p[0] for p in non_empty]
    ics = [p[1] for p in non_empty]
    st_cells = [(ir, ic) for ir in range(ROWS) for ic in range(COLS) if raw[ir][ic].strip().upper() == 'ST']
    if len(st_cells) != 1:
        raise RuntimeError(f'expected exactly one ST cell in {MAP_CSV}, found {len(st_cells)}')
    return raw, land_info, (min(irs), max(irs)), (min(ics), max(ics)), st_cells[0]


def _fill_mask(arr, alpha):
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3
    mask = brightness >= 70
    if alpha is not None:
        mask &= alpha >= 128
    return mask


def _find_st_pixel(arr, mask):
    """Locate the ST hex via its distinctive highlight border color, then
    refine to that hex's actual fill-blob centroid so it lines up with the
    other correspondence points (which are all fill-blob centroids too)."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    c = ST_COLOR_RANGE
    st_mask = (r > c['r_lo']) & (r < c['r_hi']) & (g > c['g_lo']) & (g < c['g_hi']) & (b < c['b_hi'])
    ys, xs = np.where(st_mask)
    if len(xs) == 0:
        return None
    # The highlight color can appear elsewhere (similar terrain hues) - take
    # the largest tight cluster by finding connected components of the color
    # mask and picking the smallest-bbox one near the mask's global centroid
    # isn't robust either, so instead restrict to components of modest size
    # (a highlight ring, not a whole terrain region) closest to the mask's
    # median position.
    labeled, n = ndimage.label(st_mask)
    if n == 0:
        return None
    sizes = ndimage.sum(st_mask, labeled, range(1, n + 1))
    centers = ndimage.center_of_mass(st_mask, labeled, range(1, n + 1))
    med_x, med_y = np.median(xs), np.median(ys)
    best = None
    best_d = None
    for (cy, cx), sz in zip(centers, sizes):
        if sz < 50 or sz > 5000:
            continue
        d = (cx - med_x) ** 2 + (cy - med_y) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = (cx, cy)
    return best


def estimate_initial_affine(arr, mask, terrain_extent, st_cell):
    (ir_min, ir_max), (ic_min, ic_max) = terrain_extent
    st_ir, st_ic = st_cell

    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    mode_size = np.median(sizes[(sizes > 500)])
    lo, hi = mode_size * 0.85, mode_size * 1.15
    centers = ndimage.center_of_mass(mask, labeled, range(1, n + 1))
    pts = [(cx, cy) for (cy, cx), sz in zip(centers, sizes) if lo < sz < hi]
    print(f'  size filter [{lo:.0f}, {hi:.0f}] -> {len(pts)} rough blobs (mode size {mode_size:.0f})')

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    px_min, px_max = xs.min(), xs.max()
    py_min, py_max = ys.min(), ys.max()

    ic_min_i = ic_min * 1.5 * HEX_SIZE
    ic_max_i = ic_max * 1.5 * HEX_SIZE
    ir_min_i = ir_min * np.sqrt(3) * HEX_SIZE
    ir_max_i = ir_max * np.sqrt(3) * HEX_SIZE

    dx = (px_max - px_min) / (ic_max_i - ic_min_i) * (1.5 * HEX_SIZE)
    dy = (py_max - py_min) / (ir_max_i - ir_min_i) * (np.sqrt(3) * HEX_SIZE)

    if ST_PIXEL_OVERRIDE is not None:
        st_px = ST_PIXEL_OVERRIDE
        print(f'  using ST_PIXEL_OVERRIDE = {st_px}')
    else:
        st_px = _find_st_pixel(arr, mask)
    if st_px is None:
        raise RuntimeError(
            "Could not auto-locate the ST hex via highlight color. "
            "Pass --st-pixel X Y by hand (a pixel in the middle of the "
            "highlighted 'current team position' hex near the start point) "
            "and re-run."
        )
    st_px_x, st_px_y = st_px
    print(f'  ST hex (ir={st_ir}, ic={st_ic}) at pixel ({st_px_x:.1f}, {st_px_y:.1f})')

    x0 = st_px_x - st_ic * dx
    half = dy / 2
    y0 = st_px_y + st_ir * dy + (half if st_ic % 2 == 1 else 0)
    return x0, dx, y0, dy, half, (lo, hi)


def detect_correspondences(arr, mask, x0, dx, y0, dy, half, size_bounds):
    lo, hi = size_bounds
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    centers = ndimage.center_of_mass(mask, labeled, range(1, n + 1))
    pts = [(cx, cy) for (cy, cx), sz in zip(centers, sizes) if lo * 0.7 < sz < hi * 1.3]
    print(f'  {len(pts)} candidate hex blobs (widened size filter)')

    def fit(pts, x0, dx, y0, dy, half, tol=None):
        A_x, b_x, A_y, b_y, assigned = [], [], [], [], []
        for x, y in pts:
            ic = round((x - x0) / dx)
            h = half if (ic % 2 == 1) else 0
            ir = round((y0 - h - y) / dy)
            if not (0 <= ic <= COLS - 1 and 0 <= ir <= ROWS - 1):
                continue
            if tol is not None:
                px = x0 + ic * dx
                hh = half if ic % 2 == 1 else 0
                py = y0 - ir * dy - hh
                if abs(px - x) > tol or abs(py - y) > tol:
                    continue
            assigned.append((ir, ic, x, y))
            A_x.append([1, ic])
            b_x.append(x)
            hbit = 1 if ic % 2 == 1 else 0
            A_y.append([1, -ir, -hbit])
            b_y.append(y)
        A_x, b_x, A_y, b_y = map(np.array, (A_x, b_x, A_y, b_y))
        sol_x, *_ = np.linalg.lstsq(A_x, b_x, rcond=None)
        sol_y, *_ = np.linalg.lstsq(A_y, b_y, rcond=None)
        return sol_x, sol_y, assigned

    tol = None
    for it in range(6):
        sol_x, sol_y, assigned = fit(pts, x0, dx, y0, dy, half, tol=tol)
        x0, dx = sol_x
        y0, dy, half = sol_y
        errs_x, errs_y = [], []
        for ir, ic, x, y in assigned:
            px = x0 + ic * dx
            h = half if ic % 2 == 1 else 0
            py = y0 - ir * dy - h
            errs_x.append(px - x)
            errs_y.append(py - y)
        errs_x, errs_y = np.array(errs_x), np.array(errs_y)
        print(f'  iter {it}: n={len(assigned)} xerr std={errs_x.std():.2f} yerr std={errs_y.std():.2f}')
        tol = max(15, 3 * max(errs_x.std(), errs_y.std()))

    good = {(ir, ic): (x, y) for ir, ic, x, y in assigned}
    print(f'  {len(good)} final (ir, ic) <-> pixel correspondences, '
          f'final xerr std={errs_x.std():.2f} yerr std={errs_y.std():.2f}')
    return good, (x0, dx, y0, dy, half)


def place_whole_image(im, affine):
    """Keep the ENTIRE source image intact (no per-hex warping, no
    cropping, no transparency mask) and place it with a single rigid
    affine transform fitted from the correspondence data. Only viable
    because the fit converges tightly for this source (near-zero residual -
    see the xerr/yerr std printed above); if a future source image turns
    out to be a stitched composite again (large residual that keeps
    shrinking across iterations but never gets small), a single affine
    won't align it well end to end and the old per-hex griddata warp
    approach (see git history) would be needed instead.
    """
    x0, dx, y0, dy, half = affine
    orig_w, orig_h = im.size

    scale = min(1.0, OUTPUT_LONG_EDGE / max(im.size))
    if scale < 1.0:
        new_size = (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
        im = im.resize(new_size, Image.LANCZOS)
    out = np.array(im.convert('RGBA'))

    # Map pixel (0,0)..(orig_w,orig_h) of the ORIGINAL (pre-downsample) image
    # to this module's scaled data-coordinate space, using the same
    # column/row-step + half-row-offset structure as _center() - see
    # _map_image_extent_for_size() in hex_pathfinding_demo.py for the
    # runtime-side equivalent of this formula (kept in sync by hand).
    a = (1.5 * HEX_SIZE * X_SCALE) / dx
    b = -a * x0
    c = -(np.sqrt(3) * HEX_SIZE * Y_SCALE) / dy
    d = -c * y0
    left = b
    right = a * orig_w + b
    top = d
    bottom = c * orig_h + d
    return out, [left, right, bottom, top]


def main():
    src_path = _ARGS.src or (SRC_IMAGE_PNG if os.path.exists(SRC_IMAGE_PNG) else SRC_IMAGE_JPG)
    im = Image.open(src_path)
    if im.mode == 'RGBA':
        print(f'loading {src_path} (RGBA) ...')
        full = np.array(im).astype(np.int32)
        arr = full[:, :, :3]
        src_alpha = full[:, :, 3]
    else:
        print(f'loading {src_path} (no alpha channel, using brightness-only detection) ...')
        im = im.convert('RGB')
        arr = np.array(im).astype(np.int32)
        src_alpha = None
    print('image size', im.size)

    mask = _fill_mask(arr, src_alpha)
    raw, land_info, ir_extent, ic_extent, st_cell = _load_terrain_extent()

    print('estimating initial affine calibration...')
    x0, dx, y0, dy, half, size_bounds = estimate_initial_affine(arr, mask, (ir_extent, ic_extent), st_cell)
    print(f'  seed: X0={x0:.2f} DX={dx:.2f} Y0={y0:.2f} DY={dy:.2f} HALF={half:.2f}')

    _correspondences, fitted_affine = detect_correspondences(arr, mask, x0, dx, y0, dy, half, size_bounds)

    # Per-hex griddata warping + real-terrain masking (see git history for
    # that version) was cutting the outer rim off edge-of-island hexes and,
    # per user feedback, is more aggressive than wanted anyway - keep the
    # WHOLE source image intact instead (no cropping, no transparency) and
    # place it with the single rigid affine fitted above. Only safe because
    # the fit converged to a sub-pixel residual for this source (see the
    # "final xerr/yerr std" line printed above) - a stitched/composite
    # screenshot would need the per-hex warp instead.
    out, extent = place_whole_image(im, fitted_affine)

    Image.fromarray(out, 'RGBA').save(OUTPUT_PNG)
    print(f'saved {OUTPUT_PNG}  size={out.shape[1]}x{out.shape[0]}')

    with open(EXTENT_JSON, 'w') as f:
        json.dump({'extent': extent}, f)
    print(f'saved {EXTENT_JSON}  extent={extent}')


if __name__ == '__main__':
    main()
