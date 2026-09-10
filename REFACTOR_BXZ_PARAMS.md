# BXZ Buff Parameters Refactoring (2026-09-10)

## Summary
成功将BXZ buff地块的关键参数从硬编码转移到`landInfo.json`文件中，提高了配置灵活性。

## Changes Made

### 1. landInfo.json 更新
为B、X、Z地块添加了以下新参数：

**B地块 (粮食折扣):**
- `food_discount_rate`: 0.4 (40%粮食折扣)
- `effective_steps`: 5/8/10 (B1/B2/B3分别为5/8/10步)

**X地块 (迅速移动):**
- `has_no_step_cost`: true (无需消耗步数)
- `effective_steps`: 5/8/10 (X1/X2/X3分别为5/8/10步)

**Z地块 (奖励增益):**
- `reward_bonus_rate`: 1.4 (40%奖励增益)
- `effective_steps`: 5/8/10 (Z1/Z2/Z3分别为5/8/10步)

### 2. hex_pathfinding_demo.py 更新

#### 新增功能
- 添加 `_load_buff_parameters()` 函数来从landInfo.json加载BXZ参数
- 定义全局常数 `B_DISCOUNT_MAP`, `X_BONUS_MAP`, `Z_BONUS_MAP`

#### 更新的函数
- `_apply_b_discount()`: 现在从landInfo读取折扣率而不是硬编码0.4
- `_apply_challenge_discounts()`: 现在从landInfo读取B折扣率
- `_apply_z_bonus()`: 现在从landInfo读取奖励增益率
- `_compute_bxz_adjustments()`: 现在使用全局常数，并跟踪激活的地块类型以获取正确的参数

#### 替换的硬编码值
- 所有 `{'B1': 5, 'B2': 8, 'B3': 10}` 替换为 `B_DISCOUNT_MAP`
- 所有 `{'X1': 5, 'X2': 8, 'X3': 10}` 替换为 `X_BONUS_MAP`
- 所有 `{'Z1': 5, 'Z2': 8, 'Z3': 10}` 替换为 `Z_BONUS_MAP`
- 所有硬编码的 `0.4` (40%折扣) 替换为从landInfo读取的值
- 所有硬编码的 `1.4` (140%奖励) 替换为从landInfo读取的值

## 优势

1. **配置集中化**: 所有BXZ参数现在集中在landInfo.json中，易于维护和修改
2. **灵活性提升**: 可以轻松调整折扣率、奖励率等参数，无需修改Python代码
3. **代码简化**: 移除了代码中的多个硬编码常数
4. **易于扩展**: 未来添加新的buff地块或修改参数变得更容易

## 验证

已创建 `verify_bxz_params.py` 脚本来验证参数是否正确加载：
- ✓ B1: food_discount_rate=0.4, effective_steps=5
- ✓ B2: food_discount_rate=0.4, effective_steps=8
- ✓ B3: food_discount_rate=0.4, effective_steps=10
- ✓ X1: has_no_step_cost=True, effective_steps=5
- ✓ X2: has_no_step_cost=True, effective_steps=8
- ✓ X3: has_no_step_cost=True, effective_steps=10
- ✓ Z1: reward_bonus_rate=1.4, effective_steps=5
- ✓ Z2: reward_bonus_rate=1.4, effective_steps=8
- ✓ Z3: reward_bonus_rate=1.4, effective_steps=10

## 文件修改列表

1. `landInfo.json` - 添加BXZ参数到相应地块
2. `hex_pathfinding_demo.py` - 更新参数加载和使用逻辑
3. `verify_bxz_params.py` - 新建验证脚本
