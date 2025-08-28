import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

def batch_rodrigues(rot_vecs):
    """
    优化的批量Rodrigues公式实现
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype
    
    # 获取角度（旋转向量的模长）
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)  # [B, 1]
    
    # 处理零旋转情况
    small_angle_mask = angle < 1e-4
    
    # 获取旋转轴（归一化）
    rot_dir = rot_vecs / angle  # [B, 3]
    
    # 三角函数
    cos_angle = torch.cos(angle)  # [B, 1]
    sin_angle = torch.sin(angle)  # [B, 1]
    
    # 单位矩阵
    I = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, 3, 3)
    
    # 对于小角度，直接返回单位矩阵加小扰动
    if small_angle_mask.any():
        small_rot = I.clone()
        # 对小角度情况使用一阶近似
        mask_expanded = small_angle_mask.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        return torch.where(mask_expanded, small_rot, I)  # 临时返回，下面会正确计算
    
    # 外积 axis ⊗ axis
    outer = torch.bmm(rot_dir.unsqueeze(2), rot_dir.unsqueeze(1))  # [B, 3, 3]
    
    # 反对称矩阵
    x, y, z = rot_dir[:, 0], rot_dir[:, 1], rot_dir[:, 2]
    K = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x
    
    # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
    # 使用 K² = outer - I 的关系简化计算
    one_minus_cos = (1 - cos_angle).unsqueeze(-1)
    sin_angle_expanded = sin_angle.unsqueeze(-1)
    
    rot_mat = I + sin_angle_expanded * K + one_minus_cos * (outer - I)
    
    return rot_mat

def simple_forward_kinematics(pose_param_16x3: torch.Tensor, rest_joints: torch.Tensor, device=None):
    """
    MANO hand forward kinematics with 16D axis-angle parameters
    
    Args:
      pose_param_16x3: [B,16,3] - 16维轴角参数
      rest_joints: [B,21,3] - 绝对rest坐标，函数会用差值得到相对偏移
      
    Returns:
      joint_positions: [B,21,3] - 21个关键点的3D坐标
    """
    if device is None:
        device = pose_param_16x3.device
    B = pose_param_16x3.shape[0]

    # 手部骨骼连接关系 (parent -> child)
    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],      # 食指
        [0, 9], [9, 10], [10, 11], [11, 12], # 中指
        [0, 13], [13, 14], [14, 15], [15, 16], # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
    ]
    
    # 构建父节点索引
    parent_idx = [-1] * 21  # -1 表示根节点
    for p, c in bones:
        parent_idx[c] = p

    # 关键映射：16维参数到21个关节的映射
    joint_to_param_idx = {
        0: 0,   # global root (腕关节)
        # 拇指（MCP, PIP, DIP）
        1: 1, 2: 2, 3: 3,
        # 食指
        5: 4, 6: 5, 7: 6,
        # 中指
        9: 7, 10: 8, 11: 9,
        # 无名指
        13: 10, 14: 11, 15: 12,
        # 小指
        17: 13, 18: 14, 19: 15,
        # 关节 4,8,12,16,20 (指尖) 保持不映射（无参数，跟随父关节）
    }

    # 初始化输出
    joint_positions = torch.zeros(B, 21, 3, device=device, dtype=rest_joints.dtype)
    transforms = [None] * 21  # 每项存 Bx3x3 的 global rotation

    # 根关节处理
    if 0 in joint_to_param_idx:
        root_local = pose_param_16x3[:, joint_to_param_idx[0], :]  # [B,3]
        root_rot = batch_rodrigues(root_local)  # [B,3,3]
    else:
        root_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

    # 根位置：旋转 rest_joints[0]
    joint_positions[:, 0, :] = torch.bmm(root_rot, rest_joints[:, 0, :].unsqueeze(-1)).squeeze(-1)
    transforms[0] = root_rot

    # 逐关节递归计算
    for j in range(1, 21):
        parent = parent_idx[j]
        parent_pos = joint_positions[:, parent, :]       # [B,3]
        parent_rot = transforms[parent]                  # [B,3,3]

        # 计算相对位置向量
        rel = rest_joints[:, j, :] - rest_joints[:, parent, :]  # [B,3]

        # 获取局部旋转
        if j in joint_to_param_idx:
            local_axis_angle = pose_param_16x3[:, joint_to_param_idx[j], :]  # [B,3]
            local_rot = batch_rodrigues(local_axis_angle)  # [B,3,3]
        else:
            # 指尖关节无参数，使用单位矩阵
            local_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

        # 计算全局旋转（用于子关节）
        global_rot = torch.bmm(parent_rot, local_rot)  # [B,3,3]
        transforms[j] = global_rot

        # 关节位置只受父关节全局旋转影响
        rotated_rel = torch.bmm(parent_rot, rel.unsqueeze(-1)).squeeze(-1)  # [B,3]
        joint_positions[:, j, :] = parent_pos + rotated_rel

    return joint_positions

def load_rest_hand_pose(file_path="../mano_hand_kpts/rest_hand_kpts_flat.xyz"):
    """
    从文件加载MANO手部静息姿态（21个关键点）
    """
    try:
        rest_joints = np.loadtxt(file_path)
        if rest_joints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {rest_joints.shape}")
        print(f"Successfully loaded rest pose from {file_path}")
        return rest_joints
    except Exception as e:
        print(f"Error loading rest pose from {file_path}: {e}")
        raise

class HandPoseDataset(Dataset):
    """
    手部姿态数据集 - 实时随机生成数据
    """
    def __init__(self, rest_joints, num_samples=10000, device='cuda'):
        self.rest_joints = torch.FloatTensor(rest_joints).to(device)
        self.num_samples = num_samples
        self.device = device
        
        print(f"Dataset initialized with {num_samples} samples for real-time generation on {device}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 每次随机生成新的轴角参数
        pose_params = torch.rand(16, 3, device=self.device) * np.pi  # [16, 3]
        
        # 扩展rest_joints
        rest_batch = self.rest_joints.unsqueeze(0)  # [1, 21, 3]
        pose_batch = pose_params.unsqueeze(0)       # [1, 16, 3]
        
        # 计算正运动学
        keypoints = simple_forward_kinematics(pose_batch, rest_batch, self.device)  # [1, 21, 3]
        
        # 转换为平坦格式
        X = keypoints.view(-1).cpu()        # [63]
        y = pose_params.view(-1).cpu()      # [48]
        
        return X, y

class IKNet(nn.Module):
    """
    手部逆运动学MLP网络
    """
    def __init__(self, input_dim=63, output_dim=48, hidden_dims=[512, 1024, 1024, 512, 256]):
        super(IKNet, self).__init__()

        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        output = self.network(x)
        
        # 将输出限制在 [0, π] 范围内
        output = torch.sigmoid(output) * np.pi
        
        return output

class HandIKTrainer:
    """
    手部逆运动学训练器
    """
    def __init__(self, model, rest_joints, device='cuda', learning_rate=0.001, loss_type='combined'):
        """
        初始化训练器
        
        Args:
            loss_type: 'keypoints_only' 或 'combined'
                - 'keypoints_only': 只使用关键点损失
                - 'combined': 使用轴损失 + 角度损失 + 0.1*关键点损失
        """
        self.model = model.to(device)
        self.rest_joints = torch.FloatTensor(rest_joints).to(device)
        self.device = device
        self.loss_type = loss_type
        
        print(f"Loss type: {loss_type}")
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.epoch_stats = []  # 存储每个epoch的详细统计信息
        
    def forward_kinematics_loss(self, pred_poses, target_keypoints):
        """
        正运动学一致性损失（关键点loss）
        """
        batch_size = pred_poses.shape[0]
        
        # 重塑预测的姿态参数
        pred_poses_reshaped = pred_poses.view(batch_size, 16, 3)
        
        # 扩展rest_joints
        rest_batch = self.rest_joints.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 计算正运动学
        pred_keypoints = simple_forward_kinematics(pred_poses_reshaped, rest_batch, self.device)
        pred_keypoints_flat = pred_keypoints.view(batch_size, -1)
        
        # 计算L2损失
        loss = torch.mean((pred_keypoints_flat - target_keypoints) ** 2)
        
        return loss, pred_keypoints_flat
    
    def axis_angle_loss(self, pred_poses, true_poses):
        """
        轴角度损失：分别计算轴方向损失和角度大小损失
        """
        batch_size = pred_poses.shape[0]
        
        # 重塑为 [batch, 16, 3]
        pred_reshaped = pred_poses.view(batch_size, 16, 3)
        true_reshaped = true_poses.view(batch_size, 16, 3)
        
        # 计算角度大小（旋转向量的模长）
        pred_angles = torch.norm(pred_reshaped + 1e-8, dim=2)  # [batch, 16]
        true_angles = torch.norm(true_reshaped + 1e-8, dim=2)  # [batch, 16]
        
        # 角度损失
        angle_loss = torch.mean((pred_angles - true_angles) ** 2)
        
        # 计算轴方向（归一化的旋转向量）
        pred_axes = pred_reshaped / (torch.norm(pred_reshaped, dim=2, keepdim=True) + 1e-8)
        true_axes = true_reshaped / (torch.norm(true_reshaped, dim=2, keepdim=True) + 1e-8)
        
        # 轴方向损失（使用余弦相似度）
        cosine_sim = torch.sum(pred_axes * true_axes, dim=2)  # [batch, 16]
        axis_loss = torch.mean((1 - cosine_sim) ** 2)  # 1-cos(θ)，当方向相同时为0
        
        return axis_loss, angle_loss
    
    def compute_keypoints_only_loss(self, pred_poses, target_keypoints):
        """
        计算只有关键点的损失
        """
        keypoint_loss, pred_keypoints = self.forward_kinematics_loss(pred_poses, target_keypoints)
        return keypoint_loss, pred_keypoints
    
    def compute_combined_loss(self, pred_poses, true_poses, target_keypoints):
        """
        计算组合损失：轴损失 + 角度损失 + 0.1 * 关键点损失
        """
        # 轴角度损失
        axis_loss, angle_loss = self.axis_angle_loss(pred_poses, true_poses)
        
        # 关键点损失
        keypoint_loss, pred_keypoints = self.forward_kinematics_loss(pred_poses, target_keypoints)
        
        # 组合损失
        total_loss = axis_loss + angle_loss + 0.1 * keypoint_loss
        
        return total_loss, axis_loss, angle_loss, keypoint_loss, pred_keypoints
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        total_keypoint_loss = 0
        
        # 添加进度条
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, (keypoints, pose_params) in enumerate(pbar):
            keypoints = keypoints.to(self.device)
            pose_params = pose_params.to(self.device)
            
            # 前向传播
            pred_poses = self.model(keypoints)
            
            # 根据损失类型计算损失
            if self.loss_type == 'keypoints_only':
                batch_loss, _ = self.compute_keypoints_only_loss(pred_poses, keypoints)
                axis_loss = angle_loss = 0.0
                keypoint_loss = batch_loss.item()
            else:  # combined loss
                batch_loss, axis_loss, angle_loss, keypoint_loss, _ = self.compute_combined_loss(
                    pred_poses, pose_params, keypoints)
                axis_loss = axis_loss.item()
                angle_loss = angle_loss.item()
                keypoint_loss = keypoint_loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            total_axis_loss += axis_loss
            total_angle_loss += angle_loss
            total_keypoint_loss += keypoint_loss
            
            # 更新进度条显示
            if self.loss_type == 'keypoints_only':
                pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'Keypoint': f'{keypoint_loss:.4f}'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'Axis': f'{axis_loss:.4f}',
                    'Angle': f'{angle_loss:.4f}',
                    'Keypoint': f'{keypoint_loss:.4f}'
                })
        
        n_batches = len(dataloader)
        return (total_loss/n_batches, total_axis_loss/n_batches, 
                total_angle_loss/n_batches, total_keypoint_loss/n_batches)
    
    def validate_with_analysis(self, dataloader):
        """验证并进行详细分析"""
        self.model.eval()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0 
        total_keypoint_loss = 0
        
        # 收集预测结果用于分析
        all_pred_poses = []
        all_true_poses = []
        all_pred_keypoints = []
        all_true_keypoints = []
        all_joint_errors = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for keypoints, pose_params in pbar:
                keypoints = keypoints.to(self.device)
                pose_params = pose_params.to(self.device)
                
                pred_poses = self.model(keypoints)
                
                # 根据损失类型计算损失
                if self.loss_type == 'keypoints_only':
                    batch_loss, pred_keypoints = self.compute_keypoints_only_loss(pred_poses, keypoints)
                    axis_loss = angle_loss = 0.0
                    keypoint_loss = batch_loss.item()
                else:  # combined loss
                    batch_loss, axis_loss, angle_loss, keypoint_loss, pred_keypoints = self.compute_combined_loss(
                        pred_poses, pose_params, keypoints)
                    axis_loss = axis_loss.item()
                    angle_loss = angle_loss.item()
                    keypoint_loss = keypoint_loss.item()
                
                total_loss += batch_loss.item()
                total_axis_loss += axis_loss
                total_angle_loss += angle_loss
                total_keypoint_loss += keypoint_loss
                
                # 收集数据用于分析
                all_pred_poses.append(pred_poses.cpu())
                all_true_poses.append(pose_params.cpu())
                all_pred_keypoints.append(pred_keypoints.cpu())
                all_true_keypoints.append(keypoints.cpu())
                
                # 计算关节角度误差
                joint_errors = torch.abs(pred_poses - pose_params).cpu()
                all_joint_errors.append(joint_errors)
                
                # 更新进度条显示
                if self.loss_type == 'keypoints_only':
                    pbar.set_postfix({
                        'Loss': f'{batch_loss.item():.4f}',
                        'Keypoint': f'{keypoint_loss:.4f}'
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f'{batch_loss.item():.4f}',
                        'Axis': f'{axis_loss:.4f}',
                        'Angle': f'{angle_loss:.4f}',
                        'Keypoint': f'{keypoint_loss:.4f}'
                    })
        
        n_batches = len(dataloader)
        avg_loss = total_loss/n_batches
        avg_axis_loss = total_axis_loss/n_batches
        avg_angle_loss = total_angle_loss/n_batches
        avg_keypoint_loss = total_keypoint_loss/n_batches
        
        # 合并所有数据
        all_pred_poses = torch.cat(all_pred_poses, dim=0)  # [N, 48]
        all_true_poses = torch.cat(all_true_poses, dim=0)  # [N, 48]
        all_pred_keypoints = torch.cat(all_pred_keypoints, dim=0)  # [N, 63]
        all_true_keypoints = torch.cat(all_true_keypoints, dim=0)  # [N, 63]
        all_joint_errors = torch.cat(all_joint_errors, dim=0)  # [N, 48]
        
        # 计算统计信息
        stats = {
            'avg_loss': avg_loss,
            'avg_axis_loss': avg_axis_loss,
            'avg_angle_loss': avg_angle_loss,
            'avg_keypoint_loss': avg_keypoint_loss,
            'pred_poses': all_pred_poses,
            'true_poses': all_true_poses,
            'pred_keypoints': all_pred_keypoints,
            'true_keypoints': all_true_keypoints,
            'joint_errors': all_joint_errors,
            'keypoint_errors': torch.abs(all_pred_keypoints - all_true_keypoints)
        }
        
        return avg_loss, avg_axis_loss, avg_angle_loss, avg_keypoint_loss, stats
    
    def plot_validation_analysis(self, stats, epoch):
        """绘制验证分析图表"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 重塑数据
        pred_poses = stats['pred_poses'].numpy()  # [N, 48]
        true_poses = stats['true_poses'].numpy()  # [N, 48]
        joint_errors = stats['joint_errors'].numpy()  # [N, 48]
        keypoint_errors = stats['keypoint_errors'].numpy().reshape(-1, 21, 3)  # [N, 21, 3]
        
        # 1. 关节角度预测散点图 (前5个关节)
        for i in range(5):
            ax = plt.subplot(3, 5, i+1)
            joint_idx = i * 3  # 每个关节取第一个轴角分量
            if joint_idx < pred_poses.shape[1]:
                plt.scatter(true_poses[:, joint_idx], pred_poses[:, joint_idx], 
                           alpha=0.6, s=10, c='blue')
                
                # 绘制理想预测线
                min_val = min(true_poses[:, joint_idx].min(), pred_poses[:, joint_idx].min())
                max_val = max(true_poses[:, joint_idx].max(), pred_poses[:, joint_idx].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                plt.xlabel(f'True Angle {i+1} (rad)')
                plt.ylabel(f'Predicted Angle {i+1} (rad)')
                plt.title(f'Joint {i+1} Angle')
                plt.grid(True, alpha=0.3)
        
        # 2. 轴方向相似性分析 (第6个位置)
        ax = plt.subplot(3, 5, 6)
        # 计算前3个关节的轴角向量相似性
        similarities = []
        for i in range(0, min(15, pred_poses.shape[1]), 3):
            if i+2 < pred_poses.shape[1]:
                pred_axis = pred_poses[:, i:i+3]
                true_axis = true_poses[:, i:i+3]
                
                # 计算余弦相似度
                pred_norm = np.linalg.norm(pred_axis, axis=1, keepdims=True) + 1e-8
                true_norm = np.linalg.norm(true_axis, axis=1, keepdims=True) + 1e-8
                
                pred_normalized = pred_axis / pred_norm
                true_normalized = true_axis / true_norm
                
                similarity = np.sum(pred_normalized * true_normalized, axis=1)
                similarities.append(similarity)
        
        if len(similarities) > 0:
            similarities = np.array(similarities).T  # [N, num_joints]
            boxplot_data = [similarities[:, i] for i in range(similarities.shape[1])]
            plt.boxplot(boxplot_data, labels=[f'Joint{i+1}' for i in range(similarities.shape[1])])
            plt.ylabel('Cosine Similarity')
            plt.title('Axis Direction Similarity')
            plt.grid(True, alpha=0.3)
        
        # 3. 关键点重建误差 (第7个位置)
        ax = plt.subplot(3, 5, 7)
        keypoint_error_norms = np.linalg.norm(keypoint_errors, axis=2)  # [N, 21]
        # 转置数据以便正确的boxplot格式 
        boxplot_data = [keypoint_error_norms[:, i] for i in range(21)]
        plt.boxplot(boxplot_data, labels=[f'J{i}' for i in range(21)])
        plt.ylabel('Reconstruction Error (units)')
        plt.title('Keypoint Reconstruction Error')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. 训练进度 (第8个位置)
        ax = plt.subplot(3, 5, 8)
        epochs = range(len(self.train_losses))
        plt.plot(epochs, self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(epochs, self.val_losses, label='Val Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 端效器位置误差分布 (第9个位置)
        ax = plt.subplot(3, 5, 9)
        # 选择指尖关节的误差 (关节4, 8, 12, 16, 20)
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertip_errors = keypoint_error_norms[:, fingertip_indices]
        fingertip_error_distances = np.linalg.norm(fingertip_errors, axis=1)
        
        plt.hist(fingertip_error_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('End Effector Error (units)')
        plt.ylabel('Frequency')
        plt.title('End Effector Position Error')
        plt.grid(True, alpha=0.3)
        
        # 6. 球形关节配置 3D可视化 (第10个位置)
        try:
            ax = fig.add_subplot(3, 5, 10, projection='3d')
            # 可视化前100个样本的前3个关节的轴角向量
            sample_size = min(100, pred_poses.shape[0])
            for joint_idx in range(min(3, pred_poses.shape[1]//3)):
                start_idx = joint_idx * 3
                if start_idx + 2 < pred_poses.shape[1]:
                    x = pred_poses[:sample_size, start_idx]
                    y = pred_poses[:sample_size, start_idx+1]
                    z = pred_poses[:sample_size, start_idx+2]
                    ax.scatter(x, y, z, alpha=0.6, s=20, label=f'Joint {joint_idx+1}')
            
            ax.set_xlabel('X Axis (rad)')
            ax.set_ylabel('Y Axis (rad)')
            ax.set_zlabel('Z Axis (rad)')
            ax.set_title('Spherical Joint Configurations')
            ax.legend()
        except Exception as e:
            # 如果3D绘图失败，绘制2D散点图
            ax = plt.subplot(3, 5, 10)
            sample_size = min(100, pred_poses.shape[0])
            if pred_poses.shape[1] >= 6:
                plt.scatter(pred_poses[:sample_size, 0], pred_poses[:sample_size, 3], 
                           alpha=0.6, s=20, c='blue', label='Joint 1 vs Joint 2')
                plt.xlabel('Joint 1 X (rad)')
                plt.ylabel('Joint 2 X (rad)')
                plt.title('Joint Configurations (2D)')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # 7-10. 关节角误差分布
        for i in range(4):
            ax = plt.subplot(3, 5, 11+i)
            joint_start = i * 12  # 每个图显示12个参数的误差
            joint_end = min(joint_start + 12, joint_errors.shape[1])
            
            if joint_start < joint_errors.shape[1]:
                joint_subset = joint_errors[:, joint_start:joint_end]
                boxplot_data = [joint_subset[:, j] for j in range(joint_subset.shape[1])]
                plt.boxplot(boxplot_data, labels=[f'P{j+joint_start}' for j in range(len(boxplot_data))])
                plt.ylabel('Angle Error (rad)')
                plt.title(f'Joint Angle Error Distribution {i+1}')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Validation Analysis - Epoch {epoch}', fontsize=16, y=0.98)
        plt.savefig(f'../results_handik/validation_analysis_combined_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存，不显示
        
        # 打印统计摘要
        print(f"\n=== Epoch {epoch} Validation Summary ===")
        print(f"Average Joint Angle Error: {joint_errors.mean():.6f} ± {joint_errors.std():.6f} rad")
        print(f"Max Joint Angle Error: {joint_errors.max():.6f} rad")
        print(f"Average Keypoint Error: {keypoint_error_norms.mean():.6f} ± {keypoint_error_norms.std():.6f}")
        print(f"Max Keypoint Error: {keypoint_error_norms.max():.6f}")
        
        if len(similarities) > 0:
            avg_similarity = np.mean(similarities)
            print(f"Average Axis Similarity: {avg_similarity:.4f}")
        
        return {
            'joint_angle_error_mean': joint_errors.mean(),
            'joint_angle_error_std': joint_errors.std(),
            'keypoint_error_mean': keypoint_error_norms.mean(),
            'keypoint_error_std': keypoint_error_norms.std(),
        }
    
    def train(self, train_loader, val_loader, epochs=200, plot_frequency=20):
        """训练模型"""
        print("Starting hand inverse kinematics training...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Loss type: {self.loss_type}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 2000
        
        # 为epochs添加进度条
        epoch_pbar = tqdm(range(epochs), desc="Training Progress")
        for epoch in epoch_pbar:
            # 训练
            train_loss, train_axis_loss, train_angle_loss, train_keypoint_loss = self.train_epoch(train_loader)
            
            # 验证 - 使用详细分析
            if epoch % plot_frequency == 0 or epoch == epochs - 1:
                val_loss, val_axis_loss, val_angle_loss, val_keypoint_loss, stats = self.validate_with_analysis(val_loader)
                
                # 绘制验证分析图
                analysis_stats = self.plot_validation_analysis(stats, epoch)
                self.epoch_stats.append(analysis_stats)
            else:
                # 快速验证，不进行详细分析
                val_loss, val_axis_loss, val_angle_loss, val_keypoint_loss = self.validate_simple(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_hand_ik_model.pth')
            else:
                patience_counter += 1
            
            # 更新epoch进度条显示
            if self.loss_type == 'keypoints_only':
                epoch_pbar.set_postfix({
                    'Train': f'{train_loss:.4f}',
                    'Val': f'{val_loss:.4f}',
                    'Best': f'{best_val_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'Patience': f'{patience_counter}/{max_patience}'
                })
            else:
                epoch_pbar.set_postfix({
                    'Train': f'{train_loss:.4f}',
                    'Val': f'{val_loss:.4f}',
                    'Best': f'{best_val_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'Patience': f'{patience_counter}/{max_patience}'
                })
            
            # 详细打印（减少频率）
            if epoch % 10 == 0 or epoch == epochs - 1:
                if self.loss_type == 'keypoints_only':
                    tqdm.write(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                              f"Keypoint={val_keypoint_loss:.6f}")
                else:
                    tqdm.write(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                              f"Axis={val_axis_loss:.6f}, Angle={val_angle_loss:.6f}, Keypoint={val_keypoint_loss:.6f}")
            
            # 早停
            if patience_counter >= max_patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_hand_ik_model.pth'))
        print("Training completed!")
    
    def validate_simple(self, dataloader):
        """简单验证，不进行详细分析"""
        self.model.eval()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        total_keypoint_loss = 0
        
        with torch.no_grad():
            for keypoints, pose_params in dataloader:
                keypoints = keypoints.to(self.device)
                pose_params = pose_params.to(self.device)
                
                pred_poses = self.model(keypoints)
                
                # 根据损失类型计算损失
                if self.loss_type == 'keypoints_only':
                    batch_loss, _ = self.compute_keypoints_only_loss(pred_poses, keypoints)
                    axis_loss = angle_loss = 0.0
                    keypoint_loss = batch_loss.item()
                else:  # combined loss
                    batch_loss, axis_loss, angle_loss, keypoint_loss, _ = self.compute_combined_loss(
                        pred_poses, pose_params, keypoints)
                    axis_loss = axis_loss.item()
                    angle_loss = angle_loss.item()
                    keypoint_loss = keypoint_loss.item()
                
                total_loss += batch_loss.item()
                total_axis_loss += axis_loss
                total_angle_loss += angle_loss
                total_keypoint_loss += keypoint_loss
        
        n_batches = len(dataloader)
        return (total_loss/n_batches, total_axis_loss/n_batches, 
                total_angle_loss/n_batches, total_keypoint_loss/n_batches)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        epochs = range(len(self.train_losses))
        
        plt.plot(epochs, self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(epochs, self.val_losses, label='Validation Loss', alpha=0.8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hand IK Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示

def main(loss_type='combined'):
    """
    主训练函数
    
    Args:
        loss_type: 'keypoints_only' 或 'combined'
            - 'keypoints_only': 只使用关键点损失 (forward kinematics loss)
            - 'combined': 使用轴损失 + 角度损失 + 0.1*关键点损失
    """
    print("=== MANO Hand Inverse Kinematics Training ===")
    print(f"Loss type: {loss_type}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载rest pose
    rest_joints = load_rest_hand_pose()
    print(f"Rest joints shape: {rest_joints.shape}")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = HandPoseDataset(rest_joints, num_samples=10000, device=device)  
    val_dataset = HandPoseDataset(rest_joints, num_samples=2000, device=device)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)  
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 创建模型
    model = IKNet(input_dim=63, output_dim=48)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器 - 指定损失类型
    trainer = HandIKTrainer(model, rest_joints, device=device, learning_rate=0.001, loss_type=loss_type)
    
    # 开始训练
    trainer.train(train_loader, val_loader, epochs=2000, plot_frequency=20)  # 每20个epoch绘制一次分析图  
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    print("\nTraining completed successfully!")
    print("Model saved as 'best_hand_ik_model.pth'")

if __name__ == "__main__":
    # 可以在这里选择损失类型
    # 'keypoints_only': 只有关键点损失
    # 'combined': 轴损失 + 角度损失 + 0.1*关键点损失
    main(loss_type='combined')  # 默认使用组合损失
