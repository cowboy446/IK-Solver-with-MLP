# 损失函数系统说明

## 概述

根据您的要求，我已经将损失函数重新设计为两种类型：

### 1. 关键点损失 (keypoints_only)
- **只有关键点损失**：使用正运动学一致性损失
- **计算方式**：通过预测的轴角参数进行正运动学计算，然后与目标关键点比较
- **适用场景**：当您只关心最终关键点位置准确性时

### 2. 组合损失 (combined) 
- **轴损失 (axis_loss)**：比较预测和真实轴角向量的方向
- **角度损失 (angle_loss)**：比较预测和真实轴角向量的大小(角度)
- **关键点损失 (keypoint_loss)**：正运动学一致性损失
- **组合方式**：`total_loss = axis_loss + angle_loss + 0.1 * keypoint_loss`

## 损失函数详细说明

### 轴损失 (Axis Loss)
```python
def axis_angle_loss(self, pred_poses, true_poses):
    # 计算轴方向（归一化的旋转向量）
    pred_axes = pred_reshaped / (torch.norm(pred_reshaped, dim=2, keepdim=True) + 1e-8)
    true_axes = true_reshaped / (torch.norm(true_reshaped, dim=2, keepdim=True) + 1e-8)
    
    # 轴方向损失（使用余弦相似度）
    cosine_sim = torch.sum(pred_axes * true_axes, dim=2)
    axis_loss = torch.mean((1 - cosine_sim) ** 2)  # 当方向相同时为0
```

### 角度损失 (Angle Loss)
```python
def axis_angle_loss(self, pred_poses, true_poses):
    # 计算角度大小（旋转向量的模长）
    pred_angles = torch.norm(pred_reshaped + 1e-8, dim=2)
    true_angles = torch.norm(true_reshaped + 1e-8, dim=2)
    
    # 角度损失
    angle_loss = torch.mean((pred_angles - true_angles) ** 2)
```

### 关键点损失 (Keypoint Loss)
```python
def forward_kinematics_loss(self, pred_poses, target_keypoints):
    # 通过正运动学计算预测的关键点
    pred_keypoints = simple_forward_kinematics(pred_poses_reshaped, rest_batch, self.device)
    
    # 计算L2损失
    loss = torch.mean((pred_keypoints_flat - target_keypoints) ** 2)
```

## 使用方法

### 方法1：直接修改main函数
```python
# 使用关键点损失
main(loss_type='keypoints_only')

# 使用组合损失
main(loss_type='combined')
```

### 方法2：使用测试脚本
```bash
# 测试关键点损失
python test_loss_types.py keypoints_only

# 测试组合损失
python test_loss_types.py combined
```

### 方法3：创建训练器时指定
```python
# 创建使用关键点损失的训练器
trainer = HandIKTrainer(model, rest_joints, device=device, loss_type='keypoints_only')

# 创建使用组合损失的训练器
trainer = HandIKTrainer(model, rest_joints, device=device, loss_type='combined')
```

## 输出变化

### 进度条显示
- **关键点损失模式**: 只显示总损失和关键点损失
- **组合损失模式**: 显示总损失、轴损失、角度损失、关键点损失

### 训练日志
- **关键点损失模式**: `Epoch XXX: Train=X.XXXX, Val=X.XXXX, Keypoint=X.XXXX`
- **组合损失模式**: `Epoch XXX: Train=X.XXXX, Val=X.XXXX, Axis=X.XXXX, Angle=X.XXXX, Keypoint=X.XXXX`

## 建议使用场景

### 使用关键点损失 (keypoints_only) 当：
- 只关心最终关键点位置的准确性
- 对轴角参数的准确性要求不高
- 希望训练更快收敛

### 使用组合损失 (combined) 当：
- 需要轴角参数的高精度
- 希望同时优化旋转方向和角度大小
- 需要更好的几何一致性

## 默认设置
默认使用组合损失 (`loss_type='combined'`)，这与您其他脚本中的设计保持一致。
