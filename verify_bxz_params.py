#!/usr/bin/env python3
"""Verify that BXZ buff parameters are correctly loaded from landInfo.json"""

import json

# Load landInfo
with open('landInfo.json') as f:
    terrain_db = json.load(f)

print("=" * 60)
print("BXZ Buff Parameters Verification")
print("=" * 60)

# Check B parameters (food discount)
print("\nB地块 (粮食折扣):")
for key in ['B1', 'B2', 'B3']:
    if key in terrain_db:
        terrain = terrain_db[key]
        discount_rate = terrain.get('food_discount_rate', 'NOT SET')
        effective_steps = terrain.get('effective_steps', 'NOT SET')
        print(f"  {key}: food_discount_rate={discount_rate}, effective_steps={effective_steps}")
    else:
        print(f"  {key}: NOT FOUND in landInfo.json")

# Check X parameters (no step cost)
print("\nX地块 (迅速移动):")
for key in ['X1', 'X2', 'X3']:
    if key in terrain_db:
        terrain = terrain_db[key]
        has_no_step_cost = terrain.get('has_no_step_cost', 'NOT SET')
        effective_steps = terrain.get('effective_steps', 'NOT SET')
        print(f"  {key}: has_no_step_cost={has_no_step_cost}, effective_steps={effective_steps}")
    else:
        print(f"  {key}: NOT FOUND in landInfo.json")

# Check Z parameters (reward bonus)
print("\nZ地块 (奖励增益):")
for key in ['Z1', 'Z2', 'Z3']:
    if key in terrain_db:
        terrain = terrain_db[key]
        bonus_rate = terrain.get('reward_bonus_rate', 'NOT SET')
        effective_steps = terrain.get('effective_steps', 'NOT SET')
        print(f"  {key}: reward_bonus_rate={bonus_rate}, effective_steps={effective_steps}")
    else:
        print(f"  {key}: NOT FOUND in landInfo.json")

print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)
