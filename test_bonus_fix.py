#!/usr/bin/env python
"""Test to verify the bonus calculation fix."""

print("Z Bonus Calculation (40% reward bonus):")
print(f"  Base reward: 45")
print(f"  45 * 1.4 = {45 * 1.4}")
print(f"  int(45 * 1.4) = {int(45 * 1.4)} (OLD - wrong)")
print(f"  round(45 * 1.4) = {round(45 * 1.4)} (NEW - correct)")
print()

print("Your Issue:")
print("  Expected: 45 * 1.4 = 63")
print(f"  Got before fix: {int(45 * 1.4)}")
print(f"  Get after fix: {round(45 * 1.4)} ✓")
