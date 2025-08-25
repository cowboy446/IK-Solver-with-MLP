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

class Vector3DToAxisAngleMLP(nn.Module):
    """
    将3D向量映射到轴角表示的MLP网络
    输入: 3D向量 [x, y, z]
    输出: 轴角表示 [axis_x, axis_y, axis_z, angle]，其中轴是单位向量，角度在[0, π]范围内
    """
    def __init__(self, hidden_sizes=[128, 256, 128, 64]):
        super(Vector3DToAxisAngleMLP, self).__init__()
        
        layers = []
        input_size = 3  # 3D向量输入
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        
        # 输出层 - 4维输出 (3维轴向量 + 1维角度)
        layers.append(nn.Linear(input_size, 4))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        raw_output = self.network(x)
        
        # 分离轴和角度
        axis_raw = raw_output[:, :3]  # 前3维是轴
        angle_raw = raw_output[:, 3:4]  # 第4维是角度
        
        # 归一化轴向量 (确保是单位向量)
        axis_norm = torch.norm(axis_raw, dim=1, keepdim=True)
        axis_norm = torch.clamp(axis_norm, min=1e-8)  # 避免除零
        axis_normalized = axis_raw / axis_norm
        
        # 将角度映射到 [0, π] 范围
        angle_normalized = torch.sigmoid(angle_raw) * np.pi
        
        return torch.cat([axis_normalized, angle_normalized], dim=1)

def vector_to_axis_angle_analytical(vectors):
    """
    解析方法：将3D向量转换为轴角表示
    这里我们定义一个映射规则：向量指向的方向作为旋转轴，向量的长度映射为旋转角度
    """
    # 归一化向量长度
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # 避免除零
    
    # 轴：归一化的向量方向
    axis = vectors / norms
    
    # 角度：向量长度映射到[0, π]范围
    # 使用tanh函数将长度映射到[0, 1]，再乘以π
    angle = np.tanh(norms.flatten()) * np.pi
    
    return np.column_stack([axis, angle.reshape(-1, 1)])

def axis_angle_to_rotation_matrix(axis_angle):
    """
    将轴角表示转换为旋转矩阵（用于验证）
    axis_angle: [axis_x, axis_y, axis_z, angle]
    """
    if isinstance(axis_angle, torch.Tensor):
        axis_angle = axis_angle.detach().cpu().numpy()
    
    axis = axis_angle[:3]
    angle = axis_angle[3]
    
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]], 
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

class AxisAngleTrainer:
    """
    训练器类
    """
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=15
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def custom_loss_function(self, y_pred, y_true):
        """
        自定义损失函数：考虑轴和角度的几何特性
        """
        # 分离预测和真实的轴角表示
        pred_axis = y_pred[:, :3]
        pred_angle = y_pred[:, 3]
        true_axis = y_true[:, :3]
        true_angle = y_true[:, 3]
        
        # 轴的损失：使用余弦相似度（因为轴方向可能相反但等价）
        axis_cosine = torch.sum(pred_axis * true_axis, dim=1)
        axis_loss = 1 - torch.abs(axis_cosine)  # 1 - |cos(θ)|，θ是两轴间夹角
        
        # 角度损失：直接MSE
        angle_loss = (pred_angle - true_angle) ** 2
        
        # 组合损失
        total_loss = torch.mean(axis_loss + 0.5 * angle_loss)
        
        return total_loss, torch.mean(axis_loss), torch.mean(angle_loss)
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            y_pred = self.model(batch_x)
            
            # 计算损失
            loss, axis_loss, angle_loss = self.custom_loss_function(y_pred, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_axis_loss += axis_loss.item()
            total_angle_loss += angle_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_axis_loss = total_axis_loss / len(dataloader)
        avg_angle_loss = total_angle_loss / len(dataloader)
        
        return avg_loss, avg_axis_loss, avg_angle_loss
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_axis_loss = 0
        total_angle_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                y_pred = self.model(batch_x)
                loss, axis_loss, angle_loss = self.custom_loss_function(y_pred, batch_y)
                
                total_loss += loss.item()
                total_axis_loss += axis_loss.item()
                total_angle_loss += angle_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_axis_loss = total_axis_loss / len(dataloader)
        avg_angle_loss = total_angle_loss / len(dataloader)
        
        return avg_loss, avg_axis_loss, avg_angle_loss
    
    def train(self, train_loader, val_loader, epochs=200):
        """训练模型"""
        print("开始训练 3D向量到轴角表示的映射...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_axis_loss, train_angle_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_axis_loss, val_angle_loss = self.evaluate(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_axis_angle_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                print(f"         Axis Loss={val_axis_loss:.6f}, Angle Loss={val_angle_loss:.6f}")
            
            # 早停
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_axis_angle_model.pth'))
        print("训练完成！")
    
    def predict(self, x):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            return self.model(x).cpu().numpy()

def generate_data(n_samples=5000):
    """生成训练数据"""
    # 生成随机3D向量
    vectors = np.random.normal(0, 2, (n_samples, 3))
    
    # 生成对应的轴角表示
    axis_angles = vector_to_axis_angle_analytical(vectors)
    
    return vectors.astype(np.float32), axis_angles.astype(np.float32)

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=128):
    """创建数据加载器"""
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_results(trainer, X_test, y_test):
    """绘制结果"""
    # 预测
    y_pred = trainer.predict(X_test)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 输入向量的3D可视化
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.scatter(X_test[:100, 0], X_test[:100, 1], X_test[:100, 2], 
               c=np.linalg.norm(X_test[:100], axis=1), cmap='viridis', alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Input 3D Vectors (first 100)')
    
    # 2. 真实轴方向的3D可视化
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2.scatter(y_test[:100, 0], y_test[:100, 1], y_test[:100, 2], 
               c=y_test[:100, 3], cmap='plasma', alpha=0.6)
    ax2.set_xlabel('Axis X')
    ax2.set_ylabel('Axis Y')
    ax2.set_zlabel('Axis Z')
    ax2.set_title('True Axis Directions')
    
    # 3. 预测轴方向的3D可视化
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    ax3.scatter(y_pred[:100, 0], y_pred[:100, 1], y_pred[:100, 2], 
               c=y_pred[:100, 3], cmap='plasma', alpha=0.6)
    ax3.set_xlabel('Axis X')
    ax3.set_ylabel('Axis Y')
    ax3.set_zlabel('Axis Z')
    ax3.set_title('Predicted Axis Directions')
    
    # 4. 角度对比
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.scatter(y_test[:, 3], y_pred[:, 3], alpha=0.6, s=10)
    ax4.plot([0, np.pi], [0, np.pi], 'r--', label='Perfect Prediction')
    ax4.set_xlabel('True Angle (rad)')
    ax4.set_ylabel('Predicted Angle (rad)')
    ax4.set_title('Angle Comparison')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 轴向量余弦相似度
    ax5 = fig.add_subplot(2, 4, 5)
    cosine_sim = np.sum(y_test[:, :3] * y_pred[:, :3], axis=1)
    ax5.hist(np.abs(cosine_sim), bins=50, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('|Cosine Similarity|')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Axis Direction Similarity')
    ax5.grid(True)
    
    # 6. 角度误差分布
    ax6 = fig.add_subplot(2, 4, 6)
    angle_error = np.abs(y_test[:, 3] - y_pred[:, 3])
    ax6.hist(angle_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax6.set_xlabel('Angle Error (rad)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Angle Error Distribution')
    ax6.grid(True)
    
    # 7. 训练损失曲线
    ax7 = fig.add_subplot(2, 4, 7)
    epochs = range(len(trainer.train_losses))
    ax7.plot(epochs, trainer.train_losses, label='Train Loss', color='blue')
    ax7.plot(epochs, trainer.val_losses, label='Val Loss', color='red')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Loss')
    ax7.set_title('Training Progress')
    ax7.legend()
    ax7.grid(True)
    
    # 8. 向量长度 vs 预测角度
    ax8 = fig.add_subplot(2, 4, 8)
    vector_norms = np.linalg.norm(X_test, axis=1)
    ax8.scatter(vector_norms, y_pred[:, 3], alpha=0.6, s=10, c='green')
    ax8.set_xlabel('Input Vector Norm')
    ax8.set_ylabel('Predicted Angle (rad)')
    ax8.set_title('Vector Norm vs Predicted Angle')
    ax8.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/axis_angle_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred

def evaluate_model(X_test, y_test, y_pred):
    """评估模型性能"""
    # 轴方向相似度
    axis_cosine = np.sum(y_test[:, :3] * y_pred[:, :3], axis=1)
    axis_similarity = np.mean(np.abs(axis_cosine))
    
    # 角度误差
    angle_mae = np.mean(np.abs(y_test[:, 3] - y_pred[:, 3]))
    angle_rmse = np.sqrt(np.mean((y_test[:, 3] - y_pred[:, 3]) ** 2))
    
    # 整体损失
    axis_loss = np.mean(1 - np.abs(axis_cosine))
    angle_loss = np.mean((y_test[:, 3] - y_pred[:, 3]) ** 2)
    total_loss = axis_loss + 0.5 * angle_loss
    
    return {
        'axis_similarity': axis_similarity,
        'angle_mae': angle_mae,
        'angle_rmse': angle_rmse,
        'axis_loss': axis_loss,
        'angle_loss': angle_loss,
        'total_loss': total_loss
    }

def main():
    """主函数"""
    print("=== 3D向量到轴角表示的MLP映射实验 ===")
    print("输入: 3D向量 [x, y, z]")
    print("输出: 轴角表示 [axis_x, axis_y, axis_z, angle]")
    print("映射规则: 向量方向→轴方向, tanh(向量长度)×π→角度")
    print()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成数据
    print("生成训练数据...")
    X_train, y_train = generate_data(n_samples=8000)
    X_val, y_val = generate_data(n_samples=2000)
    X_test, y_test = generate_data(n_samples=1000)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=256)
    
    # 创建模型
    model = Vector3DToAxisAngleMLP(hidden_sizes=[256, 512, 256, 128])
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = AxisAngleTrainer(model, learning_rate=0.001, device=device)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=250)
    
    # 预测和可视化
    print("\n绘制结果图...")
    y_pred = plot_results(trainer, X_test, y_test)
    
    # 评估
    print("评估模型性能...")
    metrics = evaluate_model(X_test, y_test, y_pred)
    
    print("\n=== 评估结果 ===")
    print(f"轴方向相似度 (|cos|): {metrics['axis_similarity']:.6f}")
    print(f"角度平均绝对误差: {metrics['angle_mae']:.6f} rad ({metrics['angle_mae']*180/np.pi:.2f}°)")
    print(f"角度均方根误差: {metrics['angle_rmse']:.6f} rad ({metrics['angle_rmse']*180/np.pi:.2f}°)")
    print(f"轴损失: {metrics['axis_loss']:.6f}")
    print(f"角度损失: {metrics['angle_loss']:.6f}")
    print(f"总损失: {metrics['total_loss']:.6f}")
    
    # 验证特殊点
    print("\n=== 特殊向量验证 ===")
    special_vectors = np.array([
        [1, 0, 0],  # x轴
        [0, 1, 0],  # y轴
        [0, 0, 1],  # z轴
        [1, 1, 1],  # 对角线
        [2, 0, 0],  # 长x轴
        [-1, 0, 0], # -x轴
        [0.5, 0.5, 0] # xy平面
    ], dtype=np.float32)
    
    special_pred = trainer.predict(special_vectors)
    special_true = vector_to_axis_angle_analytical(special_vectors)
    
    vector_names = ['[1,0,0]', '[0,1,0]', '[0,0,1]', '[1,1,1]', '[2,0,0]', '[-1,0,0]', '[0.5,0.5,0]']
    
    for i, name in enumerate(vector_names):
        pred_axis = special_pred[i, :3]
        pred_angle = special_pred[i, 3]
        true_axis = special_true[i, :3]
        true_angle = special_true[i, 3]
        
        cosine_sim = abs(np.dot(pred_axis, true_axis))
        angle_error = abs(pred_angle - true_angle)
        
        print(f"{name:>10}: 轴相似度={cosine_sim:.3f}, 角度误差={angle_error:.3f}rad ({angle_error*180/np.pi:.1f}°)")
    
    print("\n实验完成！结果图已保存为 axis_angle_results.png")

if __name__ == "__main__":
    main()
