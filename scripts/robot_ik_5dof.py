import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

class FiveAxisRobotArm:
    """
    Five-axis robot arm forward kinematics model
    Structure: Base(Z-rotation) -> Shoulder(Y-rotation) -> Elbow(Y-rotation) -> Wrist1(Y-rotation) -> Wrist2(Z-rotation)
    """
    def __init__(self, link_lengths=[1.2, 1.0, 0.8, 0.6, 0.4]):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = 5
        self.num_keypoints = 6  # Base + 5 joints
        
    def forward_kinematics(self, joint_angles):
        """
        Forward kinematics: compute keypoint global coordinates from joint angles
        joint_angles: np.array [batch_size, 5] each angle range [-π, π]
        Returns: np.array [batch_size, 6, 3] 3D coordinates of 6 keypoints (base + 5 joints)
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
            cos_total_2 = np.cos(link2_global_angle)
            sin_total_2 = np.sin(link2_global_angle)
            
            local_x3 = self.link_lengths[2] * cos_th3
            local_z3 = self.link_lengths[2] * sin_th3
            
            x3 = x2 + local_x3 * cos_total_2
            y3 = y2 + local_x3 * sin_total_2
            z3 = z2 + local_z3
            keypoints[i, 3] = [x3, y3, z3]
            
            cos_th4, sin_th4 = np.cos(angles[3]), np.sin(angles[3])
            
            link3_global_angle = angles[0] + angles[1] + angles[2]
            cos_total_3 = np.cos(link3_global_angle)
            sin_total_3 = np.sin(link3_global_angle)
            
            local_x4 = self.link_lengths[3] * cos_th4
            local_z4 = self.link_lengths[3] * sin_th4
            
            x4 = x3 + local_x4 * cos_total_3
            y4 = y3 + local_x4 * sin_total_3
            z4 = z3 + local_z4
            keypoints[i, 4] = [x4, y4, z4]
            
            cos_th5, sin_th5 = np.cos(angles[4]), np.sin(angles[4])
            
            final_x5 = x4 + self.link_lengths[4] * cos_th5 * cos_total_3
            final_y5 = y4 + self.link_lengths[4] * cos_th5 * sin_total_3
            final_z5 = z4 + self.link_lengths[4] * sin_th5
            keypoints[i, 5] = [final_x5, final_y5, final_z5]
        
        return keypoints
    
    def forward_kinematics_torch(self, joint_angles, device='cpu'):
        """
        Differentiable forward kinematics: compute keypoint global coordinates from joint angles (PyTorch version)
        joint_angles: torch.Tensor [batch_size, 5] each angle range [-π, π]
        Returns: torch.Tensor [batch_size, 6, 3] 3D coordinates of 6 keypoints (base + 5 joints)
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
        cos_th4 = torch.cos(joint_angles[:, 3])
        sin_th4 = torch.sin(joint_angles[:, 3])
        cos_th5 = torch.cos(joint_angles[:, 4])
        sin_th5 = torch.sin(joint_angles[:, 4])
        
        x1 = self.link_lengths[0] * cos_th1
        y1 = self.link_lengths[0] * sin_th1
        z1 = torch.zeros_like(x1)
        keypoints[:, 1] = torch.stack([x1, y1, z1], dim=1)
        
        local_x2 = self.link_lengths[1] * cos_th2
        local_z2 = self.link_lengths[1] * sin_th2
        
        x2 = x1 + local_x2 * cos_th1
        y2 = y1 + local_x2 * sin_th1
        z2 = z1 + local_z2
        keypoints[:, 2] = torch.stack([x2, y2, z2], dim=1)
        
        link2_global_angle = joint_angles[:, 0] + joint_angles[:, 1]
        cos_total_2 = torch.cos(link2_global_angle)
        sin_total_2 = torch.sin(link2_global_angle)
        
        local_x3 = self.link_lengths[2] * cos_th3
        local_z3 = self.link_lengths[2] * sin_th3
        
        x3 = x2 + local_x3 * cos_total_2
        y3 = y2 + local_x3 * sin_total_2
        z3 = z2 + local_z3
        keypoints[:, 3] = torch.stack([x3, y3, z3], dim=1)
        
        link3_global_angle = joint_angles[:, 0] + joint_angles[:, 1] + joint_angles[:, 2]
        cos_total_3 = torch.cos(link3_global_angle)
        sin_total_3 = torch.sin(link3_global_angle)
        
        local_x4 = self.link_lengths[3] * cos_th4
        local_z4 = self.link_lengths[3] * sin_th4
        
        x4 = x3 + local_x4 * cos_total_3
        y4 = y3 + local_x4 * sin_total_3
        z4 = z3 + local_z4
        keypoints[:, 4] = torch.stack([x4, y4, z4], dim=1)
        
        final_x5 = x4 + self.link_lengths[4] * cos_th5 * cos_total_3
        final_y5 = y4 + self.link_lengths[4] * cos_th5 * sin_total_3
        final_z5 = z4 + self.link_lengths[4] * sin_th5
        keypoints[:, 5] = torch.stack([final_x5, final_y5, final_z5], dim=1)
        
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
                [-2*np.pi/3, 2*np.pi/3],  # Joint3: ±120°
                [-np.pi/2, np.pi/2],      # Joint4: ±90°
                [-np.pi, np.pi]           # Joint5: full rotation
            ]
        
        joint_angles = np.zeros((n_samples, self.num_joints))
        for i, (min_ang, max_ang) in enumerate(angle_limits):
            joint_angles[:, i] = np.random.uniform(min_ang, max_ang, n_samples)
        
        return joint_angles

def joint_angles_to_axis_angle_5dof(joint_angles, robot_arm):
    """
    Convert joint angles to axis-angle representation for 5-DOF robot
    """
    batch_size = joint_angles.shape[0]
    axis_angles = np.zeros((batch_size, 5, 4))  # 5 joints, 4D each
    
    joint_axes = np.array([
        [0, 0, 1],  # Joint1: Z-axis rotation
        [0, 1, 0],  # Joint2: Y-axis rotation (pitch)
        [0, 1, 0],  # Joint3: Y-axis rotation (pitch)
        [0, 1, 0],  # Joint4: Y-axis rotation (pitch)
        [0, 0, 1]   # Joint5: Z-axis rotation
    ])
    
    for i in range(batch_size):
        for j in range(5):
            axis_angles[i, j, :3] = joint_axes[j]
            axis_angles[i, j, 3] = joint_angles[i, j]
    
    return axis_angles.reshape(batch_size, 20)  # flatten to 20D

class RobotInverseKinematics5DOF(nn.Module):
    """
    Five-axis robot arm inverse kinematics neural network
    Input: 3D coordinates of 5 keypoints (flattened to 15D vector, excluding fixed base)
    Output: axis-angle representation for 5 joints (4D per joint: 3D axis + 1D angle, total 20D)
    """
    def __init__(self, hidden_sizes=[512, 1024, 1024, 1024, 512, 256]):
        super(RobotInverseKinematics5DOF, self).__init__()
        
        layers = []
        input_size = 15  # 5 keypoints × 3D coordinates (excluding base)
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.15))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 20))
        
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        raw_output = self.network(x)
        
        batch_size = x.shape[0]
        raw_output = raw_output.view(batch_size, 5, 4)
        
        axes_raw = raw_output[:, :, :3]  # (batch_size, 5, 3)
        angles_raw = raw_output[:, :, 3:4]  # (batch_size, 5, 1)
        
        axes_norm = torch.norm(axes_raw, dim=2, keepdim=True)
        axes_norm = torch.clamp(axes_norm, min=1e-8)
        axes_normalized = axes_raw / axes_norm
        
        angles_normalized = torch.tanh(angles_raw) * np.pi
        
        output = torch.cat([axes_normalized, angles_normalized], dim=2)
        return output.view(batch_size, 20)

class Robot5DOFTrainer:
    """
    Five-axis robot arm inverse kinematics trainer
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
        
        pred_axis_angles = y_pred.view(batch_size, 5, 4)
        true_axis_angles = y_true.view(batch_size, 5, 4)
        
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
        """Evaluate model"""
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
        print("Training five-axis robot arm inverse kinematics network...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 40
        
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
                torch.save(self.model.state_dict(), 'best_robot_5dof_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 25 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                print(f"         Axis={val_axis_loss:.6f}, Angle={val_angle_loss:.6f}, "
                      f"Keypoints={val_keypoints_loss:.6f}")
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('best_robot_5dof_model.pth'))
        print("Training completed!")
    
    def predict(self, keypoints):
        """Predict"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(keypoints, np.ndarray):
                keypoints = torch.FloatTensor(keypoints).to(self.device)
            return self.model(keypoints).cpu().numpy()

def generate_robot_5dof_data(robot_arm, n_samples=10000):
    """Generate five-axis robot arm training data"""
    print(f"Generating {n_samples} five-axis robot poses...")
    
    joint_angles = robot_arm.sample_joint_angles(n_samples)
    
    keypoints = robot_arm.forward_kinematics(joint_angles)
    
    axis_angles = joint_angles_to_axis_angle_5dof(joint_angles, robot_arm)
    
    keypoints_movable = keypoints[:, 1:, :].reshape(n_samples, -1)
    
    return keypoints_movable.astype(np.float32), axis_angles.astype(np.float32)

def create_5dof_data_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Create data loaders for 5-DOF robot"""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_5dof_results(trainer, robot_arm, X_test, y_test):
    """Plot five-axis robot arm results"""
    y_pred = trainer.predict(X_test)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Robot arm visualization
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    sample_indices = [0, 10, 20, 30]
    colors = ['red', 'blue', 'green', 'orange']
    
    for idx, color in zip(sample_indices, colors):
        keypoints_movable = X_test[idx].reshape(5, 3)
        base_point = np.array([[0, 0, 0]])
        keypoints = np.vstack([base_point, keypoints_movable])
        
        ax1.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
                'o-', color=color, alpha=0.7, linewidth=2, markersize=6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('5-DOF Robot Arm Configurations')
    
    # 2. Joint angle predictions vs true (5 subplots)
    for joint_idx in range(5):
        ax = fig.add_subplot(3, 4, 2 + joint_idx)
        
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        
        true_angles = np.arctan2(np.sin(true_angles), np.cos(true_angles))
        pred_angles = np.arctan2(np.sin(pred_angles), np.cos(pred_angles))
        
        ax.scatter(true_angles, pred_angles, alpha=0.6, s=10, c='blue')
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('True Angle (rad)')
        ax.set_ylabel('Predicted Angle (rad)')
        ax.set_title(f'Joint {joint_idx+1} Angle')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-np.pi*1.1, np.pi*1.1])
        ax.set_ylim([-np.pi*1.1, np.pi*1.1])
    
    # 7. Joint angle error distribution
    ax7 = fig.add_subplot(3, 4, 7)
    all_angle_errors = []
    joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5']
    
    for joint_idx in range(5):
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        errors = np.abs(true_angles - pred_angles)
        errors = np.minimum(errors, 2*np.pi - errors)
        all_angle_errors.append(errors)
    
    ax7.boxplot(all_angle_errors, labels=joint_names)
    ax7.set_ylabel('Angle Error (rad)')
    ax7.set_title('Joint Angle Error Distribution')
    ax7.grid(True)
    plt.xticks(rotation=45)
    
    # 8. Forward kinematics reconstruction error
    ax8 = fig.add_subplot(3, 4, 8)
    pred_joint_angles = y_pred[:, [3, 7, 11, 15, 19]]
    pred_keypoints = robot_arm.forward_kinematics(pred_joint_angles)
    
    true_keypoints_full = np.zeros((len(X_test), 6, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]
    true_keypoints_full[:, 1:] = X_test.reshape(-1, 5, 3)
    
    reconstruction_errors = np.linalg.norm(
        pred_keypoints - true_keypoints_full, axis=2
    )
    
    keypoint_names = ['Base', 'Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5']
    error_data = [reconstruction_errors[:, i] for i in range(6)]
    ax8.boxplot(error_data, labels=keypoint_names)
    ax8.set_ylabel('Position Error (units)')
    ax8.set_title('Keypoint Reconstruction Error')
    ax8.grid(True)
    plt.xticks(rotation=45)
    
    # 9. Training loss curves
    ax9 = fig.add_subplot(3, 4, 9)
    epochs = range(len(trainer.train_losses))
    ax9.plot(epochs, trainer.train_losses, label='Train Loss', alpha=0.8)
    ax9.plot(epochs, trainer.val_losses, label='Val Loss', alpha=0.8)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Loss')
    ax9.set_title('Training Progress')
    ax9.legend()
    ax9.grid(True)
    
    # 10. End effector error analysis
    ax10 = fig.add_subplot(3, 4, 10)
    end_effector_errors = reconstruction_errors[:, -1]
    ax10.hist(end_effector_errors, bins=50, alpha=0.7, edgecolor='black')
    ax10.set_xlabel('End Effector Error (units)')
    ax10.set_ylabel('Frequency')
    ax10.set_title('End Effector Position Error')
    ax10.grid(True)
    
    # 11. Workspace visualization
    ax11 = fig.add_subplot(3, 4, 11, projection='3d')
    end_positions = true_keypoints_full[:200, -1, :]
    ax11.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
               c=end_effector_errors[:200], cmap='viridis', alpha=0.6)
    ax11.set_xlabel('X')
    ax11.set_ylabel('Y')
    ax11.set_zlabel('Z')
    ax11.set_title('5-DOF Workspace & Error')
    
    # 12. Axis similarity analysis
    ax12 = fig.add_subplot(3, 4, 12)
    axis_similarities = []
    for joint_idx in range(5):
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(cosine_sim)
    
    ax12.boxplot(axis_similarities, labels=joint_names)
    ax12.set_ylabel('|Cosine Similarity|')
    ax12.set_title('Axis Direction Similarity')
    ax12.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/robot_5dof_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, reconstruction_errors

def evaluate_5dof_model(robot_arm, X_test, y_test, y_pred):
    """Evaluate five-axis robot model performance"""
    joint_angle_errors = []
    axis_similarities = []
    
    for joint_idx in range(5):
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        angle_diff = np.abs(true_angles - pred_angles)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        joint_angle_errors.append(np.mean(angle_diff))
        
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(np.mean(cosine_sim))
    
    pred_joint_angles = y_pred[:, [3, 7, 11, 15, 19]]
    pred_keypoints = robot_arm.forward_kinematics(pred_joint_angles)
    
    true_keypoints_full = np.zeros((len(X_test), 6, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]
    true_keypoints_full[:, 1:] = X_test.reshape(-1, 5, 3)
    
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
    """Main function for 5-DOF robot arm experiment"""
    print("=== Five-axis robot arm inverse kinematics deep learning experiment ===")
    print("Input: 3D global coordinates of 5 keypoints (15D)")
    print("Output: Local axis-angle representation for 5 joints (20D)")
    print("Goal: Learn inverse kinematics mapping from end-effector pose to joint space")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    robot_arm = FiveAxisRobotArm(link_lengths=[1.2, 1.0, 0.8, 0.6, 0.4])
    print(f"Robot arm link lengths: {robot_arm.link_lengths}")
    
    print("Generating training data...")
    X_train, y_train = generate_robot_5dof_data(robot_arm, n_samples=20000)
    X_val, y_val = generate_robot_5dof_data(robot_arm, n_samples=4000)
    X_test, y_test = generate_robot_5dof_data(robot_arm, n_samples=2000)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    train_loader, val_loader = create_5dof_data_loaders(X_train, y_train, X_val, y_val, batch_size=256)
    
    model = RobotInverseKinematics5DOF(hidden_sizes=[512, 1024, 1024, 1024, 512, 256])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Robot5DOFTrainer(model, robot_arm, learning_rate=0.0015, device=device)
    
    trainer.train(train_loader, val_loader, epochs=400)
    
    print("\nPlotting results...")
    y_pred, reconstruction_errors = plot_5dof_results(trainer, robot_arm, X_test, y_test)
    
    print("Evaluating model performance...")
    metrics = evaluate_5dof_model(robot_arm, X_test, y_test, y_pred)
    
    print("\n=== Evaluation Results ===")
    joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5']
    for i, name in enumerate(joint_names):
        print(f"{name:>8} - Angle error: {metrics['joint_angle_errors'][i]:.4f} rad "
              f"({metrics['joint_angle_errors'][i]*180/np.pi:.2f}°), "
              f"Axis similarity: {metrics['axis_similarities'][i]:.4f}")
    
    print(f"\nEnd effector position error: {metrics['end_effector_error']:.4f} units")
    print(f"Average keypoint error: {metrics['avg_keypoint_error']:.4f} units")
    print(f"Maximum keypoint error: {metrics['max_keypoint_error']:.4f} units")
    
    print("\n=== Complexity Analysis ===")
    print(f"Input dimension: 15D (5 movable joints × 3D coordinates)")
    print(f"Output dimension: 20D (5 joints × 4D axis-angle)")
    print(f"Model complexity: Much higher than 3-DOF due to increased joint interactions")
    print(f"Expected challenges: More complex workspace, higher risk of singularities")
    
    print("\nExperiment completed! Results saved as robot_5dof_results.png")

if __name__ == "__main__":
    main()
