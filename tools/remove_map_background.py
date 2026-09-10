"""Turns a game map screenshot's flat background color transparent.

    cd tools
    python remove_map_background.py ../S25_map_rectified.jpg ../S25_map.png --bg 4C4A55

Pixels within --soft-lo of the background color (RGB distance) become fully
transparent, pixels beyond --soft-hi stay opaque, and the band in between
ramps linearly so anti-aliased hex edges (and JPEG ringing around them)
don't get a hard jagged cut. Feed the resulting RGBA PNG to
rectify_map_image.py, which keeps the alpha channel through to the
rectified output.
"""
import argparse

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('src')
    p.add_argument('out')
    p.add_argument('--bg', default='4C4A55', help='background color as RRGGBB hex (default 4C4A55)')
    p.add_argument('--soft-lo', type=float, default=6.0)
    p.add_argument('--soft-hi', type=float, default=18.0)
    args = p.parse_args()

    bg = np.array([int(args.bg.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
    rgb = np.asarray(Image.open(args.src).convert('RGB')).astype(np.float32)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    alpha = np.clip((dist - args.soft_lo) / (args.soft_hi - args.soft_lo), 0.0, 1.0)

    out = np.dstack((rgb, alpha * 255.0)).round().astype(np.uint8)
    Image.fromarray(out, 'RGBA').save(args.out)
    print(f'saved {args.out}  size={out.shape[1]}x{out.shape[0]}  '
          f'transparent={100 * (alpha == 0).mean():.1f}%  opaque={100 * (alpha == 1).mean():.1f}%')


if __name__ == '__main__':
    main()
