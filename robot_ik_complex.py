import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# 为了避免中文字体问题，使用英文标签
plt.rcParams['axes.unicode_minus'] = False

class ThreeAxisRobotArm:
    """
    三轴机械臂的正运动学模型
    关节顺序: Joint1(θ1) -> Joint2(θ2) -> Joint3(θ3)
    """
    def __init__(self, link_lengths=[1.0, 0.8, 0.6]):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = 3
        self.num_keypoints = 4  # 基座 + 3个关节点
    
    def forward_kinematics(self, joint_angles):
        """
        正运动学：从关节角度计算关键点全局坐标
        joint_angles: [θ1, θ2, θ3] 每个角度范围 [-π, π]
        返回: 4个关键点的3D坐标 (基座 + 3个关节)
        """
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.reshape(1, -1)
        
        batch_size = joint_angles.shape[0]
        keypoints = np.zeros((batch_size, self.num_keypoints, 3))
        
        for i in range(batch_size):
            angles = joint_angles[i]
            
            # 基座位置 (固定在原点)
            keypoints[i, 0] = [0, 0, 0]
            
            # 第一个关节 - 从第一个轴角计算
            # 假设第一个关节可以绕任意轴旋转
            cos_th1, sin_th1 = np.cos(angles[0]), np.sin(angles[0])
            
            # 第一段连杆：从基座到第一个关节
            # 假设第一个关节绕Z轴旋转，连杆沿局部X轴方向
            x1 = self.link_lengths[0] * cos_th1
            y1 = self.link_lengths[0] * sin_th1
            z1 = 0
            keypoints[i, 1] = [x1, y1, z1]
            
            # 第二个关节 - 在第一个关节基础上
            cos_th2, sin_th2 = np.cos(angles[1]), np.sin(angles[1])
            
            # 第二段连杆在局部坐标系中的位置（绕Y轴旋转）
            # 局部坐标系：X轴沿第一段连杆方向，Y轴垂直向上，Z轴右手坐标系
            local_x2 = self.link_lengths[1] * cos_th2
            local_z2 = self.link_lengths[1] * sin_th2
            
            # 转换到全局坐标系
            # 第一段连杆的方向向量
            link1_dir_x = cos_th1
            link1_dir_y = sin_th1
            
            # 第二段连杆的全局位置
            x2 = x1 + local_x2 * link1_dir_x
            y2 = y1 + local_x2 * link1_dir_y
            z2 = z1 + local_z2
            keypoints[i, 2] = [x2, y2, z2]
            
            # 第三个关节 - 在第二个关节基础上
            cos_th3, sin_th3 = np.cos(angles[2]), np.sin(angles[2])
            
            # 计算第二段连杆的方向
            link2_global_angle = angles[0] + angles[1]  # 累积角度
            cos_total, sin_total = np.cos(link2_global_angle), np.sin(link2_global_angle)
            
            # 第三段连杆在局部坐标系中的位置
            local_x3 = self.link_lengths[2] * cos_th3
            local_z3 = self.link_lengths[2] * sin_th3
            
            # 转换到全局坐标系
            x3 = x2 + local_x3 * cos_total
            y3 = y2 + local_x3 * sin_total
            z3 = z2 + local_z3
            keypoints[i, 3] = [x3, y3, z3]
        
        return keypoints
    
    def forward_kinematics_torch(self, joint_angles, device='cpu'):
        """
        可微分的正运动学：从关节角度计算关键点全局坐标 (PyTorch版本)
        joint_angles: torch.Tensor [batch_size, 3] 每个角度范围 [-π, π]
        返回: torch.Tensor [batch_size, 4, 3] 4个关键点的3D坐标 (基座 + 3个关节)
        """
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
        
        batch_size = joint_angles.shape[0]
        keypoints = torch.zeros((batch_size, self.num_keypoints, 3), device=device)
        
        # 基座位置 (固定在原点)
        keypoints[:, 0] = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
        
        # 批量计算三角函数
        cos_th1 = torch.cos(joint_angles[:, 0])
        sin_th1 = torch.sin(joint_angles[:, 0])
        cos_th2 = torch.cos(joint_angles[:, 1])
        sin_th2 = torch.sin(joint_angles[:, 1])
        cos_th3 = torch.cos(joint_angles[:, 2])
        sin_th3 = torch.sin(joint_angles[:, 2])
        
        # 第一个关节 - 从第一个轴角计算
        x1 = self.link_lengths[0] * cos_th1
        y1 = self.link_lengths[0] * sin_th1
        z1 = torch.zeros_like(x1)
        keypoints[:, 1] = torch.stack([x1, y1, z1], dim=1)
        
        # 第二个关节
        local_x2 = self.link_lengths[1] * cos_th2
        local_z2 = self.link_lengths[1] * sin_th2
        
        x2 = x1 + local_x2 * cos_th1
        y2 = y1 + local_x2 * sin_th1
        z2 = z1 + local_z2
        keypoints[:, 2] = torch.stack([x2, y2, z2], dim=1)
        
        # 第三个关节
        link2_global_angle = joint_angles[:, 0] + joint_angles[:, 1]
        cos_total = torch.cos(link2_global_angle)
        sin_total = torch.sin(link2_global_angle)
        
        local_x3 = self.link_lengths[2] * cos_th3
        local_z3 = self.link_lengths[2] * sin_th3
        
        x3 = x2 + local_x3 * cos_total
        y3 = y2 + local_x3 * sin_total
        z3 = z2 + local_z3
        keypoints[:, 3] = torch.stack([x3, y3, z3], dim=1)
        
        return keypoints
    
    def sample_joint_angles(self, n_samples, angle_limits=None):
        """
        采样随机关节角度
        angle_limits: 每个关节的角度限制 [[min1,max1], [min2,max2], ...]
        """
        if angle_limits is None:
            # 默认角度限制
            angle_limits = [
                [-np.pi, np.pi],          # Joint1: 完整旋转
                [-np.pi/2, np.pi/2],      # Joint2: ±90°
                [-2*np.pi/3, 2*np.pi/3]   # Joint3: ±120°
            ]
        
        joint_angles = np.zeros((n_samples, self.num_joints))
        for i, (min_ang, max_ang) in enumerate(angle_limits):
            joint_angles[:, i] = np.random.uniform(min_ang, max_ang, n_samples)
        
        return joint_angles

class RobotInverseKinematicsNet(nn.Module):
    """
    机械臂逆运动学神经网络
    输入: 3个关键点的3D坐标 (展平为9维向量，不包含固定的基座)
    输出: 3个关节的轴角表示 (每个关节4维: 3维轴 + 1维角度，共12维)
    """
    def __init__(self, hidden_sizes=[256, 512, 512, 256, 128]):
        super(RobotInverseKinematicsNet, self).__init__()
        
        layers = []
        input_size = 9  # 3个关键点 × 3D坐标 (不包含基座)
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.15))
            input_size = hidden_size
        
        # 输出层 - 12维输出 (3个关节 × 4维轴角)
        layers.append(nn.Linear(input_size, 12))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        raw_output = self.network(x)
        
        # 重新整形为 (batch_size, 3, 4) - 3个关节，每个4维
        batch_size = x.shape[0]
        raw_output = raw_output.view(batch_size, 3, 4)
        
        # 分离轴和角度
        axes_raw = raw_output[:, :, :3]  # (batch_size, 3, 3)
        angles_raw = raw_output[:, :, 3:4]  # (batch_size, 3, 1)
        
        # 归一化轴向量
        axes_norm = torch.norm(axes_raw, dim=2, keepdim=True)
        axes_norm = torch.clamp(axes_norm, min=1e-8)
        axes_normalized = axes_raw / axes_norm
        
        # 将角度映射到 [-π, π] 范围
        angles_normalized = torch.tanh(angles_raw) * np.pi
        
        # 重新组合
        output = torch.cat([axes_normalized, angles_normalized], dim=2)
        
        return output.view(batch_size, 12)  # 展平输出

def joint_angles_to_axis_angle(joint_angles, robot_arm):
    """
    将关节角度转换为轴角表示
    对于机械臂，每个关节都有固定的旋转轴
    """
    batch_size = joint_angles.shape[0]
    axis_angles = np.zeros((batch_size, 3, 4))  # 3个关节，每个4维
    
    # 定义每个关节的旋转轴（在局部坐标系中）
    joint_axes = np.array([
        [0, 0, 1],  # Joint1: Z轴旋转
        [0, 1, 0],  # Joint2: Y轴旋转 (俯仰)
        [0, 1, 0]   # Joint3: Y轴旋转 (俯仰)
    ])
    
    for i in range(batch_size):
        for j in range(3):
            axis_angles[i, j, :3] = joint_axes[j]  # 轴方向
            axis_angles[i, j, 3] = joint_angles[i, j]  # 角度
    
    return axis_angles.reshape(batch_size, 12)  # 展平

class RobotIKTrainer:
    """
    机械臂逆运动学训练器
    """
    def __init__(self, model, robot_arm, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.robot_arm = robot_arm
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=15
        )
        
        self.train_losses = []
        self.val_losses = []
        self.forward_errors = []  # 正运动学重建误差
    
    def custom_loss_function(self, y_pred, y_true, keypoints_true):
        """
        自定义损失函数：结合轴角损失和正运动学一致性损失
        """
        batch_size = y_pred.shape[0]
        
        # 重新整形
        pred_axis_angles = y_pred.view(batch_size, 3, 4)
        true_axis_angles = y_true.view(batch_size, 3, 4)
        
        # 1. 轴角损失
        pred_axes = pred_axis_angles[:, :, :3]
        pred_angles = pred_axis_angles[:, :, 3]
        true_axes = true_axis_angles[:, :, :3]
        true_angles = true_axis_angles[:, :, 3]
        
        # 轴的损失：余弦相似度
        axis_cosine = torch.sum(pred_axes * true_axes, dim=2)
        axis_loss = torch.mean(1 - torch.abs(axis_cosine))
        
        # 角度损失：MSE with periodic boundary
        angle_diff = pred_angles - true_angles
        # 处理角度的周期性
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = torch.mean(angle_diff ** 2)
        
        # 2. 正运动学一致性损失 - 使用可微分的PyTorch版本
        # 从预测的轴角计算关节角度，然后进行正运动学
        pred_keypoints = self.robot_arm.forward_kinematics_torch(pred_angles, self.device)
        
        # 重建关键点的误差 (只比较3个可动关节点，不包含基座)
        keypoints_loss = torch.mean((pred_keypoints[:, 1:, :].view(batch_size, -1) - 
                                   keypoints_true.view(batch_size, -1)) ** 2)
        
        # 组合损失
        total_loss = axis_loss + angle_loss + 0.1 * keypoints_loss
        
        return total_loss, axis_loss, angle_loss, keypoints_loss
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        total_keypoints_loss = 0
        
        for batch_keypoints, batch_axis_angles in dataloader:
            batch_keypoints = batch_keypoints.to(self.device)
            batch_axis_angles = batch_axis_angles.to(self.device)
            
            # 前向传播
            y_pred = self.model(batch_keypoints)
            
            # 计算损失
            loss, axis_loss, angle_loss, keypoints_loss = self.custom_loss_function(
                y_pred, batch_axis_angles, batch_keypoints
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_axis_loss += axis_loss.item()
            total_angle_loss += angle_loss.item()
            total_keypoints_loss += keypoints_loss.item()
        
        n_batches = len(dataloader)
        return (total_loss/n_batches, total_axis_loss/n_batches, 
                total_angle_loss/n_batches, total_keypoints_loss/n_batches)
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        total_keypoints_loss = 0
        
        with torch.no_grad():
            for batch_keypoints, batch_axis_angles in dataloader:
                batch_keypoints = batch_keypoints.to(self.device)
                batch_axis_angles = batch_axis_angles.to(self.device)
                
                y_pred = self.model(batch_keypoints)
                loss, axis_loss, angle_loss, keypoints_loss = self.custom_loss_function(
                    y_pred, batch_axis_angles, batch_keypoints
                )
                
                total_loss += loss.item()
                total_axis_loss += axis_loss.item()
                total_angle_loss += angle_loss.item()
                total_keypoints_loss += keypoints_loss.item()
        
        n_batches = len(dataloader)
        return (total_loss/n_batches, total_axis_loss/n_batches, 
                total_angle_loss/n_batches, total_keypoints_loss/n_batches)
    
    def train(self, train_loader, val_loader, epochs=300):
        """训练模型"""
        print("开始训练四轴机械臂逆运动学网络...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 40
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_axis_loss, train_angle_loss, train_keypoints_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_axis_loss, val_angle_loss, val_keypoints_loss = self.evaluate(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.forward_errors.append(val_keypoints_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_robot_ik_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 25 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                print(f"         Axis={val_axis_loss:.6f}, Angle={val_angle_loss:.6f}, "
                      f"Keypoints={val_keypoints_loss:.6f}")
            
            # 早停
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_robot_ik_model.pth'))
        print("训练完成！")
    
    def predict(self, keypoints):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(keypoints, np.ndarray):
                keypoints = torch.FloatTensor(keypoints).to(self.device)
            return self.model(keypoints).cpu().numpy()

def generate_robot_data(robot_arm, n_samples=10000):
    """生成机械臂训练数据"""
    print(f"生成 {n_samples} 个机械臂姿态...")
    
    # 采样关节角度
    joint_angles = robot_arm.sample_joint_angles(n_samples)
    
    # 计算正运动学得到关键点
    keypoints = robot_arm.forward_kinematics(joint_angles)
    
    # 转换为轴角表示
    axis_angles = joint_angles_to_axis_angle(joint_angles, robot_arm)
    
    # 只取可动关节点 (不包含基座，只要3个关节点) (n_samples, 9)
    keypoints_movable = keypoints[:, 1:, :].reshape(n_samples, -1)
    
    return keypoints_movable.astype(np.float32), axis_angles.astype(np.float32)

def create_robot_data_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    """创建数据加载器"""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_robot_results(trainer, robot_arm, X_test, y_test):
    """绘制机械臂结果"""
    # 预测
    y_pred = trainer.predict(X_test)
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 机械臂可视化 (选几个样本)
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    sample_indices = [0, 10, 20]
    colors = ['red', 'blue', 'green']
    
    for idx, color in zip(sample_indices, colors):
        # 重建完整的关键点（包含基座）
        keypoints_movable = X_test[idx].reshape(3, 3)  # 3个可动关节点
        base_point = np.array([[0, 0, 0]])  # 基座点
        keypoints = np.vstack([base_point, keypoints_movable])  # 4个点
        
        ax1.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
                'o-', color=color, alpha=0.7, linewidth=2, markersize=6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Robot Arm Configurations')
    
    # 2. 预测vs真实关节角度对比 (3个子图)
    for joint_idx in range(3):
        ax = fig.add_subplot(3, 4, 2 + joint_idx)
        
        # 提取第joint_idx个关节的角度 (在轴角表示中是第4个元素)
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        
        # 处理角度周期性 - 将角度映射到 [-π, π] 范围
        true_angles = np.arctan2(np.sin(true_angles), np.cos(true_angles))
        pred_angles = np.arctan2(np.sin(pred_angles), np.cos(pred_angles))
        
        ax.scatter(true_angles, pred_angles, alpha=0.6, s=10, c='blue')
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # 添加角度周期性的等价线
        ax.plot([-np.pi, np.pi], [np.pi, -np.pi], 'orange', alpha=0.5, linestyle=':', label='Periodic Equivalent')
        
        ax.set_xlabel('True Angle (rad)')
        ax.set_ylabel('Predicted Angle (rad)')
        ax.set_title(f'Joint {joint_idx+1} Angle')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-np.pi*1.1, np.pi*1.1])
        ax.set_ylim([-np.pi*1.1, np.pi*1.1])
        
        if joint_idx == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=8)
    
    # 3. 关节角度误差分布
    ax5 = fig.add_subplot(3, 4, 5)
    all_angle_errors = []
    joint_names = ['Joint1', 'Joint2', 'Joint3']
    
    for joint_idx in range(3):
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        errors = np.abs(true_angles - pred_angles)
        errors = np.minimum(errors, 2*np.pi - errors)  # 处理角度周期性
        all_angle_errors.append(errors)
    
    ax5.boxplot(all_angle_errors, labels=joint_names)
    ax5.set_ylabel('Angle Error (rad)')
    ax5.set_title('Joint Angle Error Distribution')
    ax5.grid(True)
    
    # 4. 正运动学重建误差
    ax6 = fig.add_subplot(3, 4, 6)
    # 从预测的轴角计算关节角度
    pred_joint_angles = y_pred[:, [3, 7, 11]]  # 提取角度
    pred_keypoints = robot_arm.forward_kinematics(pred_joint_angles)
    
    # 重建完整的关键点进行比较
    true_keypoints_full = np.zeros((len(X_test), 4, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]  # 基座
    true_keypoints_full[:, 1:] = X_test.reshape(-1, 3, 3)  # 可动关节
    
    # 计算重建误差
    reconstruction_errors = np.linalg.norm(
        pred_keypoints - true_keypoints_full, axis=2
    )  # (n_samples, 4)
    
    keypoint_names = ['Base', 'Joint1', 'Joint2', 'Joint3']
    # 转置数据以匹配boxplot的期望格式
    error_data = [reconstruction_errors[:, i] for i in range(4)]
    ax6.boxplot(error_data, labels=keypoint_names)
    ax6.set_ylabel('Position Error (units)')
    ax6.set_title('Keypoint Reconstruction Error')
    ax6.grid(True)
    
    # 5. 训练损失曲线
    ax7 = fig.add_subplot(3, 4, 7)
    epochs = range(len(trainer.train_losses))
    ax7.plot(epochs, trainer.train_losses, label='Train Loss', alpha=0.8)
    ax7.plot(epochs, trainer.val_losses, label='Val Loss', alpha=0.8)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Loss')
    ax7.set_title('Training Progress')
    ax7.legend()
    ax7.grid(True)
    
    # 6. 各类损失分解
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.plot(epochs, trainer.forward_errors, label='Keypoints Loss', color='orange')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Forward Kinematics Error')
    ax8.set_title('Reconstruction Error Progress')
    ax8.legend()
    ax8.grid(True)
    
    # 7. 末端位置误差分析
    ax9 = fig.add_subplot(3, 4, 9)
    end_effector_errors = reconstruction_errors[:, -1]  # 末端执行器误差
    ax9.hist(end_effector_errors, bins=50, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('End Effector Error (units)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('End Effector Position Error')
    ax9.grid(True)
    
    # 工作空间可视化
    ax10 = fig.add_subplot(3, 4, 10, projection='3d')
    end_positions = true_keypoints_full[:200, -1, :]  # 末端位置
    ax10.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
               c=end_effector_errors[:200], cmap='viridis', alpha=0.6)
    ax10.set_xlabel('X')
    ax10.set_ylabel('Y')
    ax10.set_zlabel('Z')
    ax10.set_title('Workspace & Error Distribution')
    
    # 轴相似度分析
    ax11 = fig.add_subplot(3, 4, 11)
    axis_similarities = []
    for joint_idx in range(3):
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(cosine_sim)
    
    ax11.boxplot(axis_similarities, labels=joint_names)
    ax11.set_ylabel('|Cosine Similarity|')
    ax11.set_title('Axis Direction Similarity')
    ax11.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/robot_ik_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, reconstruction_errors

def evaluate_robot_model(robot_arm, X_test, y_test, y_pred):
    """评估机械臂模型性能"""
    # 关节角度误差
    joint_angle_errors = []
    axis_similarities = []
    
    for joint_idx in range(3):
        # 角度误差
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        angle_diff = np.abs(true_angles - pred_angles)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # 处理周期性
        joint_angle_errors.append(np.mean(angle_diff))
        
        # 轴相似度
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(np.mean(cosine_sim))
    
    # 正运动学重建误差
    pred_joint_angles = y_pred[:, [3, 7, 11]]
    pred_keypoints = robot_arm.forward_kinematics(pred_joint_angles)
    
    # 重建完整的关键点进行比较
    true_keypoints_full = np.zeros((len(X_test), 4, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]  # 基座
    true_keypoints_full[:, 1:] = X_test.reshape(-1, 3, 3)  # 可动关节
    
    reconstruction_errors = np.linalg.norm(pred_keypoints - true_keypoints_full, axis=2)
    end_effector_error = np.mean(reconstruction_errors[:, -1])
    avg_keypoint_error = np.mean(reconstruction_errors)
    
    return {
        'joint_angle_errors': joint_angle_errors,
        'axis_similarities': axis_similarities,
        'end_effector_error': end_effector_error,
        'avg_keypoint_error': avg_keypoint_error,
        'max_keypoint_error': np.max(reconstruction_errors)
    }

def main():
    """主函数"""
    print("=== 三轴机械臂逆运动学深度学习实验 ===")
    print("输入: 3个关键点的3D全局坐标 (9维)")
    print("输出: 3个关节的局部轴角表示 (12维)")
    print("目标: 学习从末端位姿到关节空间的逆运动学映射")
    print()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建机械臂模型
    robot_arm = ThreeAxisRobotArm(link_lengths=[1.2, 1.0, 0.8])
    print(f"机械臂连杆长度: {robot_arm.link_lengths}")
    
    # 生成数据
    print("生成训练数据...")
    X_train, y_train = generate_robot_data(robot_arm, n_samples=15000)
    X_val, y_val = generate_robot_data(robot_arm, n_samples=3000)
    X_test, y_test = generate_robot_data(robot_arm, n_samples=2000)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建数据加载器
    train_loader, val_loader = create_robot_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=512
    )
    
    # 创建模型
    model = RobotInverseKinematicsNet(hidden_sizes=[512, 1024, 1024, 512, 256])
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = RobotIKTrainer(model, robot_arm, learning_rate=0.0015, device=device)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=400)
    
    # 预测和可视化
    print("\n绘制结果图...")
    y_pred, reconstruction_errors = plot_robot_results(trainer, robot_arm, X_test, y_test)
    
    # 评估
    print("评估模型性能...")
    metrics = evaluate_robot_model(robot_arm, X_test, y_test, y_pred)
    
    print("\n=== 评估结果 ===")
    joint_names = ['Joint1', 'Joint2', 'Joint3']
    for i, name in enumerate(joint_names):
        print(f"{name:>8} - 角度误差: {metrics['joint_angle_errors'][i]:.4f} rad "
              f"({metrics['joint_angle_errors'][i]*180/np.pi:.2f}°), "
              f"轴相似度: {metrics['axis_similarities'][i]:.4f}")
    
    print(f"\n末端执行器位置误差: {metrics['end_effector_error']:.4f} units")
    print(f"平均关键点误差: {metrics['avg_keypoint_error']:.4f} units")
    print(f"最大关键点误差: {metrics['max_keypoint_error']:.4f} units")
    
    print("\n=== 关节影响分析 ===")
    # 分析每个关节对关键点位置的影响程度
    base_config = np.array([0, 0, 0])  # 基准配置
    base_keypoints = robot_arm.forward_kinematics(base_config.reshape(1, -1))[0]
    
    joint_names = ['Joint1', 'Joint2', 'Joint3']
    perturbations = [0.1, 0.1, 0.1]  # 每个关节扰动0.1弧度
    
    print("基准关键点位置:")
    for i, pos in enumerate(base_keypoints):
        print(f"  关键点{i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("\n各关节扰动对关键点位置的影响:")
    for joint_idx in range(3):
        perturbed_config = base_config.copy()
        perturbed_config[joint_idx] += perturbations[joint_idx]
        perturbed_keypoints = robot_arm.forward_kinematics(perturbed_config.reshape(1, -1))[0]
        
        print(f"\n{joint_names[joint_idx]} 关节扰动 {perturbations[joint_idx]:.1f} 弧度后:")
        total_change = 0
        for i, (orig, new) in enumerate(zip(base_keypoints, perturbed_keypoints)):
            change = np.linalg.norm(new - orig)
            total_change += change
            print(f"  关键点{i} 变化: {change:.4f} units")
        print(f"  总变化量: {total_change:.4f} units")
    
    print("\n=== 多解性分析 ===")
    # 检查是否真的存在多解
    # 取一个特定的关键点配置，看看训练数据中是否有重复
    test_keypoint = X_test[0].reshape(1, -1)  # 取第一个测试样本
    
    # 在训练数据中查找相同的关键点配置
    tolerance = 1e-4
    similar_indices = []
    for i in range(min(1000, len(X_train))):  # 检查前1000个样本
        diff = np.linalg.norm(X_train[i] - test_keypoint[0])
        if diff < tolerance:
            similar_indices.append(i)
    
    print(f"在训练数据中找到 {len(similar_indices)} 个相似的关键点配置 (tolerance={tolerance})")
    
    if len(similar_indices) > 1:
        print("对应的关节角度:")
        for idx in similar_indices[:5]:  # 只显示前5个
            angles = y_train[idx, [3, 7, 11, 15]]  # 提取角度
            print(f"  样本 {idx}: [{angles[0]:.4f}, {angles[1]:.4f}, {angles[2]:.4f}, {angles[3]:.4f}]")
    
    # 测试特殊配置
    print("\n=== 特殊配置测试 ===")
    special_configs = np.array([
        [0, 0, 0],                    # 零位
        [np.pi/2, 0, 0],             # Joint1 90°
        [0, np.pi/4, 0],             # Joint2 45°
        [0, 0, np.pi/3],             # Joint3 60°
        [np.pi/4, np.pi/6, -np.pi/4] # 复合姿态
    ])
    
    config_names = ['Zero', 'Joint1_90°', 'Joint2_45°', 'Joint3_60°', 'Complex']
    
    for i, (config, name) in enumerate(zip(special_configs, config_names)):
        # 正运动学
        true_keypoints = robot_arm.forward_kinematics(config.reshape(1, -1))
        true_axis_angles = joint_angles_to_axis_angle(config.reshape(1, -1), robot_arm)
        
        # 预测 - 只使用可动关节点
        keypoints_movable = true_keypoints[:, 1:, :].reshape(1, -1)  # 去掉基座，只要3个关节点
        pred_axis_angles = trainer.predict(keypoints_movable)
        
        # 重建误差
        pred_joints = pred_axis_angles[0, [3, 7, 11]]
        pred_keypoints = robot_arm.forward_kinematics(pred_joints.reshape(1, -1))
        reconstruction_error = np.linalg.norm(pred_keypoints - true_keypoints)
        
        print(f"{name:>10}: 重建误差 = {reconstruction_error:.4f} units")
    
    print("\n实验完成！结果图已保存为 robot_ik_results.png")

if __name__ == "__main__":
    main()
