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

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix
    axis_angle: [batch_size, 4] where first 3 are axis (normalized), last is angle
    Returns: [batch_size, 3, 3] rotation matrices
    """
    batch_size = axis_angle.shape[0]
    device = axis_angle.device if isinstance(axis_angle, torch.Tensor) else 'cpu'
    
    if isinstance(axis_angle, np.ndarray):
        axis_angle = torch.FloatTensor(axis_angle)
    
    axes = axis_angle[:, :3]  # [batch_size, 3]
    angles = axis_angle[:, 3]  # [batch_size]
    
    # Normalize axes
    axes_norm = torch.norm(axes, dim=1, keepdim=True)
    axes_norm = torch.clamp(axes_norm, min=1e-8)
    axes = axes / axes_norm
    
    # Rodrigues' rotation formula
    cos_angle = torch.cos(angles)  # [batch_size]
    sin_angle = torch.sin(angles)  # [batch_size]
    one_minus_cos = 1 - cos_angle  # [batch_size]
    
    # Cross product matrix for each axis
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    
    # Identity matrix
    I = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Skew-symmetric matrix
    K = torch.zeros(batch_size, 3, 3, device=device)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x
    
    # Outer product of axis with itself
    outer = torch.bmm(axes.unsqueeze(2), axes.unsqueeze(1))  # [batch_size, 3, 3]
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    # But K² = axis⊗axis - I, so R = I + sin(θ)K + (1-cos(θ))(axis⊗axis - I)
    # R = cos(θ)I + sin(θ)K + (1-cos(θ))axis⊗axis
    R = (cos_angle.unsqueeze(1).unsqueeze(2) * I + 
         sin_angle.unsqueeze(1).unsqueeze(2) * K + 
         one_minus_cos.unsqueeze(1).unsqueeze(2) * outer)
    
    return R

class SphericalJointRobotArm:
    """
    Multi-DOF robot arm with spherical joints (3 DOF per joint)
    Each joint can rotate freely around any axis
    """
    def __init__(self, link_lengths=[1.2, 1.0, 0.8, 0.6, 0.4], num_joints=5):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = num_joints
        self.num_keypoints = num_joints + 1  # Base + joints
        
    def forward_kinematics(self, joint_axis_angles):
        """
        Forward kinematics for spherical joints
        joint_axis_angles: np.array [batch_size, num_joints, 4] 
                          Each joint has 4D: [axis_x, axis_y, axis_z, angle]
        Returns: np.array [batch_size, num_keypoints, 3] 3D coordinates
        """
        if joint_axis_angles.ndim == 2:
            joint_axis_angles = joint_axis_angles.reshape(1, self.num_joints, 4)
        
        batch_size = joint_axis_angles.shape[0]
        keypoints = np.zeros((batch_size, self.num_keypoints, 3))
        
        for i in range(batch_size):
            # Base is always at origin
            keypoints[i, 0] = [0, 0, 0]
            
            # Current position and orientation
            current_pos = np.array([0, 0, 0])
            current_orientation = np.eye(3)  # Identity matrix
            
            for joint_idx in range(self.num_joints):
                # Get current joint's axis-angle
                axis_angle = joint_axis_angles[i, joint_idx]
                
                # Convert to rotation matrix
                R = axis_angle_to_rotation_matrix(
                    torch.FloatTensor(axis_angle).unsqueeze(0)
                ).numpy()[0]
                
                # Update orientation: new_orientation = current_orientation @ R
                current_orientation = current_orientation @ R
                
                # Move along the link in the current orientation
                # Assume each link points in local +X direction initially
                link_vector = np.array([self.link_lengths[joint_idx], 0, 0])
                global_link_vector = current_orientation @ link_vector
                
                # Update position
                current_pos = current_pos + global_link_vector
                
                # Store keypoint
                keypoints[i, joint_idx + 1] = current_pos.copy()
        
        return keypoints
    
    def forward_kinematics_torch(self, joint_axis_angles, device='cpu'):
        """
        Differentiable forward kinematics for spherical joints (PyTorch version)
        joint_axis_angles: torch.Tensor [batch_size, num_joints, 4]
        Returns: torch.Tensor [batch_size, num_keypoints, 3]
        """
        if joint_axis_angles.dim() == 2:
            joint_axis_angles = joint_axis_angles.unsqueeze(0)
        
        batch_size = joint_axis_angles.shape[0]
        keypoints = torch.zeros((batch_size, self.num_keypoints, 3), device=device)
        
        # Base is always at origin
        keypoints[:, 0] = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
        
        # Current position and orientation for each sample in batch
        current_pos = torch.zeros((batch_size, 3), device=device)
        current_orientation = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for joint_idx in range(self.num_joints):
            # Get current joint's axis-angle for all samples
            axis_angle = joint_axis_angles[:, joint_idx, :]  # [batch_size, 4]
            
            # Convert to rotation matrix
            R = axis_angle_to_rotation_matrix(axis_angle)  # [batch_size, 3, 3]
            
            # Update orientation: new_orientation = current_orientation @ R
            current_orientation = torch.bmm(current_orientation, R)
            
            # Move along the link in the current orientation
            # Assume each link points in local +X direction initially
            link_vector = torch.tensor([self.link_lengths[joint_idx], 0, 0], 
                                     device=device, dtype=torch.float32)
            link_vector = link_vector.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2)  # [batch_size, 3, 1]
            
            # Transform link vector to global frame
            global_link_vector = torch.bmm(current_orientation, link_vector).squeeze(2)  # [batch_size, 3]
            
            # Update position
            current_pos = current_pos + global_link_vector
            
            # Store keypoint
            keypoints[:, joint_idx + 1] = current_pos.clone()
        
        return keypoints
    
    def sample_joint_axis_angles(self, n_samples, angle_limits=None):
        """
        Sample random axis-angle representations for spherical joints
        """
        if angle_limits is None:
            # Default angle limits for each joint
            angle_limits = [
                [-np.pi, np.pi],      # Joint1: full rotation
                [-np.pi/2, np.pi/2],  # Joint2: ±90°
                [-np.pi/2, np.pi/2],  # Joint3: ±90°
                [-np.pi/2, np.pi/2],  # Joint4: ±90°
                [-np.pi, np.pi]       # Joint5: full rotation
            ]
        
        joint_axis_angles = np.zeros((n_samples, self.num_joints, 4))
        
        for i in range(n_samples):
            for j in range(self.num_joints):
                # Random axis (normalized)
                axis = np.random.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                
                # Random angle within limits
                min_ang, max_ang = angle_limits[j]
                angle = np.random.uniform(min_ang, max_ang)
                
                joint_axis_angles[i, j] = np.concatenate([axis, [angle]])
        
        return joint_axis_angles

class SphericalJointInverseKinematics(nn.Module):
    """
    Neural network for spherical joint robot arm inverse kinematics
    Input: 3D coordinates of keypoints (flattened, excluding base)
    Output: axis-angle representation for spherical joints
    """
    def __init__(self, num_joints=5, hidden_sizes=[512, 1024, 1024, 1024, 512, 256]):
        super(SphericalJointInverseKinematics, self).__init__()
        
        self.num_joints = num_joints
        input_size = num_joints * 3  # keypoints × 3D coordinates (excluding base)
        output_size = num_joints * 4  # joints × 4D axis-angle
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.15))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        raw_output = self.network(x)
        
        batch_size = x.shape[0]
        raw_output = raw_output.view(batch_size, self.num_joints, 4)
        
        # Separate axes and angles
        axes_raw = raw_output[:, :, :3]  # [batch_size, num_joints, 3]
        angles_raw = raw_output[:, :, 3:4]  # [batch_size, num_joints, 1]
        
        # Normalize axes
        axes_norm = torch.norm(axes_raw, dim=2, keepdim=True)
        axes_norm = torch.clamp(axes_norm, min=1e-8)
        axes_normalized = axes_raw / axes_norm
        
        # Normalize angles to [-π, π]
        angles_normalized = torch.tanh(angles_raw) * np.pi
        
        # Combine normalized axes and angles
        output = torch.cat([axes_normalized, angles_normalized], dim=2)
        
        return output.view(batch_size, self.num_joints * 4)

class SphericalJointTrainer:
    """
    Trainer for spherical joint robot arm
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
        Custom loss function for spherical joints
        """
        batch_size = y_pred.shape[0]
        num_joints = self.robot_arm.num_joints
        
        # Reshape predictions and targets
        pred_axis_angles = y_pred.view(batch_size, num_joints, 4)
        true_axis_angles = y_true.view(batch_size, num_joints, 4)
        
        pred_axes = pred_axis_angles[:, :, :3]  # [batch_size, num_joints, 3]
        pred_angles = pred_axis_angles[:, :, 3]  # [batch_size, num_joints]
        true_axes = true_axis_angles[:, :, :3]
        true_angles = true_axis_angles[:, :, 3]
        
        # Axis direction loss (cosine similarity)
        axis_cosine = torch.sum(pred_axes * true_axes, dim=2)  # [batch_size, num_joints]
        axis_loss = torch.mean(1 - torch.abs(axis_cosine))
        
        # Angle difference loss (handle periodicity)
        angle_diff = pred_angles - true_angles
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = torch.mean(angle_diff ** 2)
        
        # Forward kinematics consistency loss
        pred_keypoints = self.robot_arm.forward_kinematics_torch(
            pred_axis_angles, self.device
        )
        
        # Exclude base point (index 0) for loss calculation
        keypoints_loss = torch.mean(
            (pred_keypoints[:, 1:, :].view(batch_size, -1) - 
             keypoints_true.view(batch_size, -1)) ** 2
        )
        
        # Combined loss
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
            
            # Gradient clipping
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
        print("Training spherical joint robot arm inverse kinematics network...")
        
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
                torch.save(self.model.state_dict(), 'best_spherical_robot_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 25 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                print(f"         Axis={val_axis_loss:.6f}, Angle={val_angle_loss:.6f}, "
                      f"Keypoints={val_keypoints_loss:.6f}")
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_spherical_robot_model.pth'))
        print("Training completed!")
    
    def predict(self, keypoints):
        """Predict axis-angles from keypoints"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(keypoints, np.ndarray):
                keypoints = torch.FloatTensor(keypoints).to(self.device)
            return self.model(keypoints).cpu().numpy()

def generate_spherical_robot_data(robot_arm, n_samples=10000):
    """Generate training data for spherical joint robot"""
    print(f"Generating {n_samples} spherical joint robot poses...")
    
    # Sample random axis-angles for all joints
    joint_axis_angles = robot_arm.sample_joint_axis_angles(n_samples)
    
    # Compute forward kinematics
    keypoints = robot_arm.forward_kinematics(joint_axis_angles)
    
    # Prepare training data
    # X: keypoints excluding base (flattened)
    # y: joint axis-angles (flattened)
    X = keypoints[:, 1:, :].reshape(n_samples, -1)  # Exclude base point
    y = joint_axis_angles.reshape(n_samples, -1)    # Flatten axis-angles
    
    return X.astype(np.float32), y.astype(np.float32)

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Create PyTorch data loaders"""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_spherical_results(trainer, robot_arm, X_test, y_test):
    """Plot results for spherical joint robot"""
    y_pred = trainer.predict(X_test)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Robot arm visualization
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    sample_indices = [0, 10, 20, 30]
    colors = ['red', 'blue', 'green', 'orange']
    
    for idx, color in zip(sample_indices, colors):
        keypoints_movable = X_test[idx].reshape(robot_arm.num_joints, 3)
        base_point = np.array([[0, 0, 0]])
        keypoints = np.vstack([base_point, keypoints_movable])
        
        ax1.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
                'o-', color=color, alpha=0.7, linewidth=2, markersize=6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Spherical Joint Robot Configurations')
    
    # 2-6. Joint angle predictions vs true for each joint
    for joint_idx in range(min(5, robot_arm.num_joints)):
        ax = fig.add_subplot(3, 4, 2 + joint_idx)
        
        true_angles = y_test[:, joint_idx*4 + 3]  # Extract angle from axis-angle
        pred_angles = y_pred[:, joint_idx*4 + 3]
        
        # Normalize angles to [-π, π]
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
    
    # 7. Axis direction similarity
    ax7 = fig.add_subplot(3, 4, 7)
    axis_similarities = []
    joint_names = [f'Joint{i+1}' for i in range(robot_arm.num_joints)]
    
    for joint_idx in range(robot_arm.num_joints):
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        
        # Compute cosine similarity
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(cosine_sim)
    
    ax7.boxplot(axis_similarities, labels=joint_names[:len(axis_similarities)])
    ax7.set_ylabel('|Cosine Similarity|')
    ax7.set_title('Axis Direction Similarity')
    ax7.grid(True)
    plt.xticks(rotation=45)
    
    # 8. Forward kinematics reconstruction error
    ax8 = fig.add_subplot(3, 4, 8)
    pred_axis_angles = y_pred.reshape(-1, robot_arm.num_joints, 4)
    pred_keypoints = robot_arm.forward_kinematics(pred_axis_angles)
    
    true_keypoints_full = np.zeros((len(X_test), robot_arm.num_keypoints, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]  # Base
    true_keypoints_full[:, 1:] = X_test.reshape(-1, robot_arm.num_joints, 3)
    
    reconstruction_errors = np.linalg.norm(
        pred_keypoints - true_keypoints_full, axis=2
    )
    
    keypoint_names = ['Base'] + [f'Joint{i+1}' for i in range(robot_arm.num_joints)]
    error_data = [reconstruction_errors[:, i] for i in range(robot_arm.num_keypoints)]
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
    ax11.set_title('Spherical Joint Workspace & Error')
    
    # 12. Angle error distribution
    ax12 = fig.add_subplot(3, 4, 12)
    all_angle_errors = []
    
    for joint_idx in range(robot_arm.num_joints):
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        errors = np.abs(true_angles - pred_angles)
        errors = np.minimum(errors, 2*np.pi - errors)  # Handle periodicity
        all_angle_errors.append(errors)
    
    ax12.boxplot(all_angle_errors, labels=joint_names)
    ax12.set_ylabel('Angle Error (rad)')
    ax12.set_title('Joint Angle Error Distribution')
    ax12.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/spherical_robot_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, reconstruction_errors

def evaluate_spherical_model(robot_arm, X_test, y_test, y_pred):
    """Evaluate spherical joint robot model performance"""
    joint_angle_errors = []
    axis_similarities = []
    
    for joint_idx in range(robot_arm.num_joints):
        # Angle errors
        true_angles = y_test[:, joint_idx*4 + 3]
        pred_angles = y_pred[:, joint_idx*4 + 3]
        angle_diff = np.abs(true_angles - pred_angles)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        joint_angle_errors.append(np.mean(angle_diff))
        
        # Axis similarities
        true_axes = y_test[:, joint_idx*4:joint_idx*4+3]
        pred_axes = y_pred[:, joint_idx*4:joint_idx*4+3]
        cosine_sim = np.abs(np.sum(true_axes * pred_axes, axis=1))
        axis_similarities.append(np.mean(cosine_sim))
    
    # Forward kinematics errors
    pred_axis_angles = y_pred.reshape(-1, robot_arm.num_joints, 4)
    pred_keypoints = robot_arm.forward_kinematics(pred_axis_angles)
    
    true_keypoints_full = np.zeros((len(X_test), robot_arm.num_keypoints, 3))
    true_keypoints_full[:, 0] = [0, 0, 0]
    true_keypoints_full[:, 1:] = X_test.reshape(-1, robot_arm.num_joints, 3)
    
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
    """Main function for spherical joint robot arm experiment"""
    print("=== Spherical Joint Robot Arm Inverse Kinematics Experiment ===")
    print("Each joint has 3 DOF (spherical joint) - can rotate around any axis")
    print("Input: 3D global coordinates of keypoints")
    print("Output: Axis-angle representation for each spherical joint")
    print("Goal: Learn inverse kinematics for truly multi-DOF robot arm")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create robot with spherical joints
    num_joints = 5
    robot_arm = SphericalJointRobotArm(
        link_lengths=[1.2, 1.0, 0.8, 0.6, 0.4], 
        num_joints=num_joints
    )
    print(f"Robot arm: {num_joints} spherical joints")
    print(f"Total DOF: {num_joints * 3} (3 DOF per joint)")
    print(f"Link lengths: {robot_arm.link_lengths}")
    
    print("Generating training data...")
    X_train, y_train = generate_spherical_robot_data(robot_arm, n_samples=20000)
    X_val, y_val = generate_spherical_robot_data(robot_arm, n_samples=4000)
    X_test, y_test = generate_spherical_robot_data(robot_arm, n_samples=2000)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=256)
    
    # Create model
    model = SphericalJointInverseKinematics(
        num_joints=num_joints,
        hidden_sizes=[512, 1024, 1024, 1024, 512, 256]
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SphericalJointTrainer(model, robot_arm, learning_rate=0.001, device=device)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=400)
    
    print("\nPlotting results...")
    y_pred, reconstruction_errors = plot_spherical_results(trainer, robot_arm, X_test, y_test)
    
    print("Evaluating model performance...")
    metrics = evaluate_spherical_model(robot_arm, X_test, y_test, y_pred)
    
    print("\n=== Evaluation Results ===")
    joint_names = [f'Joint{i+1}' for i in range(num_joints)]
    for i, name in enumerate(joint_names):
        print(f"{name:>8} - Angle error: {metrics['joint_angle_errors'][i]:.4f} rad "
              f"({metrics['joint_angle_errors'][i]*180/np.pi:.2f}°), "
              f"Axis similarity: {metrics['axis_similarities'][i]:.4f}")
    
    print(f"\nEnd effector position error: {metrics['end_effector_error']:.4f} units")
    print(f"Average keypoint error: {metrics['avg_keypoint_error']:.4f} units")
    print(f"Maximum keypoint error: {metrics['max_keypoint_error']:.4f} units")
    
    print("\n=== Complexity Analysis ===")
    print(f"Input dimension: {num_joints * 3}D ({num_joints} joints × 3D coordinates)")
    print(f"Output dimension: {num_joints * 4}D ({num_joints} joints × 4D axis-angle)")
    print(f"Total DOF: {num_joints * 3} (true multi-DOF system)")
    print(f"Expected challenges: Much higher complexity, more workspace coverage")
    
    print("\nExperiment completed! Results saved as spherical_robot_results.png")

if __name__ == "__main__":
    main()
