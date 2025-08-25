import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

class ThreeAxisRobotArm:
    """
    Three-axis robot arm forward kinematics model
    Joint order: Joint1(θ1) -> Joint2(θ2) -> Joint3(θ3)
    """
    def __init__(self, link_lengths=[1.0, 0.8, 0.6]):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = 3
        self.num_keypoints = 4
    
    def forward_kinematics(self, joint_angles):
        """
        Forward kinematics: compute keypoint global coordinates from joint angles
        joint_angles: [θ1, θ2, θ3] each angle range [-π, π]
        returns: 4 keypoints 3D coordinates (base + 3 joints)
        """
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.reshape(1, -1)
        
        batch_size = joint_angles.shape[0]
        keypoints = np.zeros((batch_size, self.num_keypoints, 3))
        
        for i in range(batch_size):
            angles = joint_angles[i]
            
            keypoints[i, 0] = [0, 0, 0]
            
            cos_th1, sin_th1 = np.cos(angles[0]), np.sin(angles[0])
            
            x1 = self.link_lengths[0] * cos_th1
            y1 = self.link_lengths[0] * sin_th1
            z1 = 0
            keypoints[i, 1] = [x1, y1, z1]
            
            cos_th2, sin_th2 = np.cos(angles[1]), np.sin(angles[1])
            
            local_x2 = self.link_lengths[1] * cos_th2
            local_z2 = self.link_lengths[1] * sin_th2
            
            link1_dir_x = cos_th1
            link1_dir_y = sin_th1
            
            x2 = x1 + local_x2 * link1_dir_x
            y2 = y1 + local_x2 * link1_dir_y
            z2 = z1 + local_z2
            keypoints[i, 2] = [x2, y2, z2]
            
            cos_th3, sin_th3 = np.cos(angles[2]), np.sin(angles[2])
            
            link2_global_angle = angles[0] + angles[1]
            cos_total, sin_total = np.cos(link2_global_angle), np.sin(link2_global_angle)
            
            local_x3 = self.link_lengths[2] * cos_th3
            local_z3 = self.link_lengths[2] * sin_th3
            
            x3 = x2 + local_x3 * cos_total
            y3 = y2 + local_x3 * sin_total
            z3 = z2 + local_z3
            keypoints[i, 3] = [x3, y3, z3]
        
        return keypoints
    
    def forward_kinematics_torch(self, joint_angles, device='cpu'):
        """
        Differentiable forward kinematics: compute keypoint global coordinates from joint angles (PyTorch version)
        joint_angles: torch.Tensor [batch_size, 3] each angle range [-π, π]
        Returns: torch.Tensor [batch_size, 4, 3] 3D coordinates of 4 keypoints (base + 3 joints)
        """
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
        
        batch_size = joint_angles.shape[0]
        keypoints = torch.zeros((batch_size, self.num_keypoints, 3), device=device)
        
        keypoints[:, 0] = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
        
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
        
        # Second joint
        local_x2 = self.link_lengths[1] * cos_th2
        local_z2 = self.link_lengths[1] * sin_th2
        
        x2 = x1 + local_x2 * cos_th1
        y2 = y1 + local_x2 * sin_th1
        z2 = z1 + local_z2
        keypoints[:, 2] = torch.stack([x2, y2, z2], dim=1)
        
        # Third joint
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
        Sample random joint angles
        angle_limits: angle limits for each joint [[min1,max1], [min2,max2], ...]
        """
        if angle_limits is None:
            angle_limits = [
                [-np.pi, np.pi],          # Joint1: full rotation
                [-np.pi/2, np.pi/2],      # Joint2: ±90°
                [-2*np.pi/3, 2*np.pi/3]   # Joint3: ±120°
            ]
        
        joint_angles = np.zeros((n_samples, self.num_joints))
        for i, (min_ang, max_ang) in enumerate(angle_limits):
            joint_angles[:, i] = np.random.uniform(min_ang, max_ang, n_samples)
        
        return joint_angles

class RobotInverseKinematicsNet(nn.Module):
    """
    Robot arm inverse kinematics neural network
    Input: 3D coordinates of 3 keypoints (flattened to 9D vector, excluding fixed base)
    Output: axis-angle representation for 3 joints (4D per joint: 3D axis + 1D angle, total 12D)
    """
    def __init__(self, hidden_sizes=[256, 512, 512, 256, 128]):
        super(RobotInverseKinematicsNet, self).__init__()
        
        layers = []
        input_size = 9  # 3 keypoints × 3D coordinates (excluding base)
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.15))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 12))
        
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        raw_output = self.network(x)
        
        batch_size = x.shape[0]
        raw_output = raw_output.view(batch_size, 3, 4)
        
        axes_raw = raw_output[:, :, :3]  # (batch_size, 3, 3)
        angles_raw = raw_output[:, :, 3:4]  # (batch_size, 3, 1)
        
        axes_norm = torch.norm(axes_raw, dim=2, keepdim=True)
        axes_norm = torch.clamp(axes_norm, min=1e-8)
        axes_normalized = axes_raw / axes_norm
        
        angles_normalized = torch.tanh(angles_raw) * np.pi
        
        output = torch.cat([axes_normalized, angles_normalized], dim=2)
        
        return output.view(batch_size, 12)  # 展平输出

def joint_angles_to_axis_angle(joint_angles, robot_arm):
    """
    将关节角度转换为轴角表示
    对于机械臂，每个关节都有固定的旋转轴
    """
    batch_size = joint_angles.shape[0]
    axis_angles = np.zeros((batch_size, 3, 4))  # 3 joints, 4D each
    
    joint_axes = np.array([
        [0, 0, 1],  # Joint1: Z-axis rotation
        [0, 1, 0],  # Joint2: Y-axis rotation (pitch)
        [0, 1, 0]   # Joint3: Y-axis rotation (pitch)
    ])
    
    for i in range(batch_size):
        for j in range(3):
            axis_angles[i, j, :3] = joint_axes[j]  # axis direction
            axis_angles[i, j, 3] = joint_angles[i, j]  # angle
    
    return axis_angles.reshape(batch_size, 12)  # flatten

class RobotIKTrainer:
    """
    Robot arm inverse kinematics trainer
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
        self.forward_errors = []
    
    def custom_loss_function(self, y_pred, y_true, keypoints_true):
        """
        Custom loss function: combine axis-angle loss and forward kinematics consistency loss
        """
        batch_size = y_pred.shape[0]
        
        pred_axis_angles = y_pred.view(batch_size, 3, 4)
        true_axis_angles = y_true.view(batch_size, 3, 4)
        
        pred_axes = pred_axis_angles[:, :, :3]
        pred_angles = pred_axis_angles[:, :, 3]
        true_axes = true_axis_angles[:, :, :3]
        true_angles = true_axis_angles[:, :, 3]
        
        axis_cosine = torch.sum(pred_axes * true_axes, dim=2)
        axis_loss = torch.mean(1 - torch.abs(axis_cosine))
        
        angle_diff = pred_angles - true_angles
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = torch.mean(angle_diff ** 2)
        
        pred_keypoints = self.robot_arm.forward_kinematics_torch(pred_angles, self.device)
        
        keypoints_loss = torch.mean((pred_keypoints[:, 1:, :].view(batch_size, -1) - 
                                   keypoints_true.view(batch_size, -1)) ** 2)
        
        total_loss = axis_loss + angle_loss + 0.1 * keypoints_loss
        
        return total_loss, axis_loss, angle_loss, keypoints_loss
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        total_keypoints_loss = 0
        
        for batch_keypoints, batch_axis_angles in dataloader:
            batch_keypoints = batch_keypoints.to(self.device)
            batch_axis_angles = batch_axis_angles.to(self.device)
            
            y_pred = self.model(batch_keypoints)
            
            loss, axis_loss, angle_loss, keypoints_loss = self.custom_loss_function(
                y_pred, batch_axis_angles, batch_keypoints
            )
            
            self.optimizer.zero_grad()
            loss.backward()

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
        """Train model"""
        print("Training three-axis robot arm inverse kinematics network...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 400
        
        for epoch in range(epochs):
            train_loss, train_axis_loss, train_angle_loss, train_keypoints_loss = self.train_epoch(train_loader)
            
            val_loss, val_axis_loss, val_angle_loss, val_keypoints_loss = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.forward_errors.append(val_keypoints_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_robot_ik_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 25 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                print(f"         Axis={val_axis_loss:.6f}, Angle={val_angle_loss:.6f}, "
                      f"Keypoints={val_keypoints_loss:.6f}")
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('best_robot_ik_model.pth'))
        print("Training completed!")
    
    def predict(self, keypoints):
        """Predict"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(keypoints, np.ndarray):
                keypoints = torch.FloatTensor(keypoints).to(self.device)
            return self.model(keypoints).cpu().numpy()

def generate_robot_data(robot_arm, n_samples=10000):
    """Generate robot arm training data"""
    print(f"Generating {n_samples} robot poses...")
    
    joint_angles = robot_arm.sample_joint_angles(n_samples)
    
    keypoints = robot_arm.forward_kinematics(joint_angles)
    
    axis_angles = joint_angles_to_axis_angle(joint_angles, robot_arm)
    
    keypoints_movable = keypoints[:, 1:, :].reshape(n_samples, -1)
    
    return keypoints_movable.astype(np.float32), axis_angles.astype(np.float32)

def create_robot_data_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Create data loaders"""
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
    """Plot robot arm results"""
    y_pred = trainer.predict(X_test)
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    sample_indices = [0, 10, 20]
    colors = ['red', 'blue', 'green']
    
    for idx, color in zip(sample_indices, colors):
        keypoints_movable = X_test[idx].reshape(3, 3)  # 3 movable joint points
        base_point = np.array([[0, 0, 0]])  # base point
        keypoints = np.vstack([base_point, keypoints_movable])  # 4 points
        
        ax1.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
                'o-', color=color, alpha=0.7, linewidth=2, markersize=6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Robot Arm Configurations')
    
    for joint_idx in range(3):
        ax = fig.add_subplot(3, 4, 2 + joint_idx)
        
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        
        true_angles = np.arctan2(np.sin(true_angles), np.cos(true_angles))
        pred_angles = np.arctan2(np.sin(pred_angles), np.cos(pred_angles))
        
        ax.scatter(true_angles, pred_angles, alpha=0.6, s=10, c='blue')
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax.plot([-np.pi, np.pi], [np.pi, -np.pi], 'orange', alpha=0.5, linestyle=':', label='Periodic Equivalent')
        
        ax.set_xlabel('True Angle (rad)')
        ax.set_ylabel('Predicted Angle (rad)')
        ax.set_title(f'Joint {joint_idx+1} Angle')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-np.pi*1.1, np.pi*1.1])
        ax.set_ylim([-np.pi*1.1, np.pi*1.1])
        
        if joint_idx == 0:
            ax.legend(fontsize=8)
    
    ax5 = fig.add_subplot(3, 4, 5)
    all_angle_errors = []
    joint_names = ['Joint1', 'Joint2', 'Joint3']
    
    for joint_idx in range(3):
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        errors = np.abs(true_angles - pred_angles)
        errors = np.minimum(errors, 2*np.pi - errors)
        all_angle_errors.append(errors)
    
    ax5.boxplot(all_angle_errors, labels=joint_names)
    ax5.set_ylabel('Angle Error (rad)')
    ax5.set_title('Joint Angle Error Distribution')
    ax5.grid(True)
    
    ax6 = fig.add_subplot(3, 4, 6)
    pred_joint_angles = y_pred[:, [3, 7, 11]]
    pred_keypoints = robot_arm.forward_kinematics(pred_joint_angles)
    
    true_keypoints_full = np.zeros((len(X_test), 4, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]
    true_keypoints_full[:, 1:] = X_test.reshape(-1, 3, 3)
    
    reconstruction_errors = np.linalg.norm(
        pred_keypoints - true_keypoints_full, axis=2
    )
    
    keypoint_names = ['Base', 'Joint1', 'Joint2', 'Joint3']
    error_data = [reconstruction_errors[:, i] for i in range(4)]
    ax6.boxplot(error_data, labels=keypoint_names)
    ax6.set_ylabel('Position Error (units)')
    ax6.set_title('Keypoint Reconstruction Error')
    ax6.grid(True)
    
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
    
    ax9 = fig.add_subplot(3, 4, 9)
    end_effector_errors = reconstruction_errors[:, -1]
    ax9.hist(end_effector_errors, bins=50, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('End Effector Error (units)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('End Effector Position Error')
    ax9.grid(True)
    
    ax10 = fig.add_subplot(3, 4, 10, projection='3d')
    end_positions = true_keypoints_full[:200, -1, :]
    ax10.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
               c=end_effector_errors[:200], cmap='viridis', alpha=0.6)
    ax10.set_xlabel('X')
    ax10.set_ylabel('Y')
    ax10.set_zlabel('Z')
    ax10.set_title('Workspace & Error Distribution')
    
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
    """Evaluate robot model performance"""
    joint_angle_errors = []
    axis_similarities = []
    
    for joint_idx in range(3):
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        angle_diff = np.abs(true_angles - pred_angles)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        joint_angle_errors.append(np.mean(angle_diff))
        
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(np.mean(cosine_sim))
    
    pred_joint_angles = y_pred[:, [3, 7, 11]]
    pred_keypoints = robot_arm.forward_kinematics(pred_joint_angles)
    
    true_keypoints_full = np.zeros((len(X_test), 4, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]
    true_keypoints_full[:, 1:] = X_test.reshape(-1, 3, 3)
    
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
    """Main function"""
    print("=== Three-axis robot arm inverse kinematics deep learning experiment ===")
    print("Input: 3D global coordinates of 3 keypoints (9D)")
    print("Output: Local axis-angle representation for 3 joints (12D)")
    print("Goal: Learn inverse kinematics mapping from end-effector pose to joint space")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    robot_arm = ThreeAxisRobotArm(link_lengths=[1.2, 1.0, 0.8])
    print(f"Robot arm link lengths: {robot_arm.link_lengths}")
    
    print("Generating training data...")
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
    model = RobotInverseKinematicsNet(hidden_sizes=[512, 1024, 1024, 512, 256])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = RobotIKTrainer(model, robot_arm, learning_rate=0.0015, device=device)
    
    trainer.train(train_loader, val_loader, epochs=1000)
    
    print("\nPlotting results...")
    y_pred, reconstruction_errors = plot_robot_results(trainer, robot_arm, X_test, y_test)
    
    print("Evaluating model performance...")
    metrics = evaluate_robot_model(robot_arm, X_test, y_test, y_pred)
    
    print("\n=== Evaluation Results ===")
    joint_names = ['Joint1', 'Joint2', 'Joint3']
    for i, name in enumerate(joint_names):
        print(f"{name:>8} - Angle error: {metrics['joint_angle_errors'][i]:.4f} rad "
              f"({metrics['joint_angle_errors'][i]*180/np.pi:.2f}°), "
              f"Axis similarity: {metrics['axis_similarities'][i]:.4f}")
    
    print(f"\nEnd effector position error: {metrics['end_effector_error']:.4f} units")
    print(f"Average keypoint error: {metrics['avg_keypoint_error']:.4f} units")
    print(f"Maximum keypoint error: {metrics['max_keypoint_error']:.4f} units")
    
    print("\n=== Joint Influence Analysis ===")
    base_config = np.array([0, 0, 0])
    base_keypoints = robot_arm.forward_kinematics(base_config.reshape(1, -1))[0]
    
    joint_names = ['Joint1', 'Joint2', 'Joint3']
    perturbations = [0.1, 0.1, 0.1]
    
    print("Base keypoint positions:")
    for i, pos in enumerate(base_keypoints):
        print(f"  Keypoint{i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("\nJoint perturbation effects on keypoint positions:")
    for joint_idx in range(3):
        perturbed_config = base_config.copy()
        perturbed_config[joint_idx] += perturbations[joint_idx]
        perturbed_keypoints = robot_arm.forward_kinematics(perturbed_config.reshape(1, -1))[0]
        
        print(f"\n{joint_names[joint_idx]} joint perturbed by {perturbations[joint_idx]:.1f} radians:")
        total_change = 0
        for i, (orig, new) in enumerate(zip(base_keypoints, perturbed_keypoints)):
            change = np.linalg.norm(new - orig)
            total_change += change
            print(f"  Keypoint{i} change: {change:.4f} units")
        print(f"  Total change: {total_change:.4f} units")
    
    print("\n=== Special Configuration Test ===")
    special_configs = np.array([
        [0, 0, 0],
        [np.pi/2, 0, 0],
        [0, np.pi/4, 0],
        [0, 0, np.pi/3],
        [np.pi/4, np.pi/6, -np.pi/4]
    ])
    
    config_names = ['Zero', 'Joint1_90°', 'Joint2_45°', 'Joint3_60°', 'Complex']
    
    for i, (config, name) in enumerate(zip(special_configs, config_names)):
        true_keypoints = robot_arm.forward_kinematics(config.reshape(1, -1))
        true_axis_angles = joint_angles_to_axis_angle(config.reshape(1, -1), robot_arm)
        
        keypoints_movable = true_keypoints[:, 1:, :].reshape(1, -1)
        pred_axis_angles = trainer.predict(keypoints_movable)
        
        pred_joints = pred_axis_angles[0, [3, 7, 11]]
        pred_keypoints = robot_arm.forward_kinematics(pred_joints.reshape(1, -1))
        reconstruction_error = np.linalg.norm(pred_keypoints - true_keypoints)
        
        print(f"{name:>10}: Reconstruction error = {reconstruction_error:.4f} units")
    
    print("\nExperiment completed! Results saved as robot_ik_results.png")

if __name__ == "__main__":
    main()
