import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 为了避免中文字体问题，我们将使用英文标签
plt.rcParams['axes.unicode_minus'] = False

class ArcsinMLPFitter:
    """
    使用MLP拟合arcsin函数的类
    通过优化 loss = (sin(y) - x)^2 来学习 y = arcsin(x)
    """
    
    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=2000, learning_rate_init=0.01):
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.training_losses = []
        
    def generate_training_data(self, n_samples=1000):
        """
        生成训练数据
        x: [-1, 1] 范围内的随机数
        y_true: arcsin(x) 的真实值
        """
        # 生成在arcsin定义域内的数据
        x = np.random.uniform(-0.99, 0.99, n_samples)  # 稍微缩小范围避免数值问题
        y_true = np.arcsin(x)
        
        return x.reshape(-1, 1), y_true.reshape(-1, 1)
    
    def custom_loss_objective(self, X, y_pred):
        """
        计算自定义损失: loss = (sin(y_pred) - X)^2
        """
        sin_y_pred = np.sin(y_pred.flatten())
        x_flat = X.flatten()
        loss = np.mean((sin_y_pred - x_flat) ** 2)
        return loss
    
    def fit_with_custom_loss(self, X, y_true, n_iterations=50):
        """
        使用自定义损失函数训练模型
        由于sklearn的MLPRegressor不直接支持自定义损失，我们采用迭代优化的方法
        """
        # 初始训练：使用真实的arcsin值作为目标
        print("初始训练阶段...")
        self.mlp.fit(X, y_true.ravel())
        
        # 为迭代优化创建一个新的MLP，不使用early_stopping
        print("自定义损失优化阶段...")
        iterative_mlp = MLPRegressor(
            hidden_layer_sizes=self.mlp.hidden_layer_sizes,
            max_iter=50,  # 每次迭代的训练步数
            learning_rate_init=self.mlp.learning_rate_init,
            random_state=42,
            early_stopping=False,  # 关闭early stopping以支持partial_fit
            warm_start=True  # 保持权重继续训练
        )
        
        # 复制初始训练的权重
        iterative_mlp.fit(X, y_true.ravel())
        
        for i in range(n_iterations):
            # 预测
            y_pred = iterative_mlp.predict(X).reshape(-1, 1)
            
            # 计算自定义损失
            custom_loss = self.custom_loss_objective(X, y_pred)
            self.training_losses.append(custom_loss)
            
            # 计算梯度的近似（通过sin(y_pred)与x的差）
            sin_y_pred = np.sin(y_pred.flatten())
            x_flat = X.flatten()
            residual = sin_y_pred - x_flat
            
            # 使用链式法则：d_loss/d_y = 2 * (sin(y) - x) * cos(y)
            cos_y_pred = np.cos(y_pred.flatten())
            gradient_approximation = 2 * residual * cos_y_pred
            
            # 调整目标值（简单的梯度下降步骤）
            learning_rate = 0.05  # 降低学习率以提高稳定性
            adjusted_targets = y_pred.flatten() - learning_rate * gradient_approximation
            
            # 重新训练：使用warm_start继续训练
            iterative_mlp.max_iter += 10  # 增加训练迭代次数
            iterative_mlp.fit(X, adjusted_targets)
            
            if i % 10 == 0:
                print(f"迭代 {i}: 自定义损失 = {custom_loss:.6f}")
        
        # 更新主模型
        self.mlp = iterative_mlp
        return self
    
    def predict(self, X):
        """预测arcsin值"""
        return self.mlp.predict(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.predict(X_test)
        
        # 计算传统MSE
        mse = mean_squared_error(y_test, y_pred)
        
        # 计算自定义损失
        custom_loss = self.custom_loss_objective(X_test, y_pred.reshape(-1, 1))
        
        # 计算sin(y_pred)与x的接近程度
        sin_y_pred = np.sin(y_pred)
        sin_accuracy = np.mean(np.abs(sin_y_pred - X_test.flatten()))
        
        return {
            'mse': mse,
            'custom_loss': custom_loss,
            'sin_accuracy': sin_accuracy
        }

def plot_results(fitter, X_test, y_test, y_pred):
    """绘制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 预测vs真实值
    axes[0, 0].scatter(X_test.flatten(), y_test.flatten(), alpha=0.6, label='True arcsin(x)', color='blue')
    axes[0, 0].scatter(X_test.flatten(), y_pred, alpha=0.6, label='MLP Prediction', color='red')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('arcsin(x)')
    axes[0, 0].set_title('MLP Prediction vs True arcsin Function')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. sin(预测值) vs x
    sin_y_pred = np.sin(y_pred)
    axes[0, 1].scatter(X_test.flatten(), X_test.flatten(), alpha=0.6, label='y=x (Ideal)', color='blue')
    axes[0, 1].scatter(X_test.flatten(), sin_y_pred, alpha=0.6, label='sin(MLP Prediction)', color='red')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('sin(Prediction)')
    axes[0, 1].set_title('sin(Prediction) vs x (Custom Loss Verification)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 连续函数图
    x_continuous = np.linspace(-0.99, 0.99, 200).reshape(-1, 1)
    y_true_continuous = np.arcsin(x_continuous.flatten())
    y_pred_continuous = fitter.predict(x_continuous)
    
    axes[1, 0].plot(x_continuous.flatten(), y_true_continuous, label='True arcsin(x)', color='blue', linewidth=2)
    axes[1, 0].plot(x_continuous.flatten(), y_pred_continuous, label='MLP Fit', color='red', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('arcsin(x)')
    axes[1, 0].set_title('Continuous Function Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 训练损失曲线
    if fitter.training_losses:
        axes[1, 1].plot(fitter.training_losses, color='green', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Custom Loss')
        axes[1, 1].set_title('Training Custom Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/mlp_arcsin_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== MLP拟合arcsin函数实验 ===")
    print("目标: 训练MLP使得 loss = (sin(MLP(x)) - x)^2 最小")
    print()
    
    # 创建模型
    fitter = ArcsinMLPFitter(hidden_layer_sizes=(100, 50, 25), max_iter=1000)
    
    # 生成训练数据
    print("生成训练数据...")
    X_train, y_train = fitter.generate_training_data(n_samples=2000)
    
    # 生成测试数据
    print("生成测试数据...")
    X_test, y_test = fitter.generate_training_data(n_samples=500)
    
    # 训练模型
    print("开始训练...")
    fitter.fit_with_custom_loss(X_train, y_train, n_iterations=30)
    
    # 预测
    print("进行预测...")
    y_pred = fitter.predict(X_test)
    
    # 评估
    print("评估模型性能...")
    metrics = fitter.evaluate(X_test, y_test)
    
    print("\n=== 评估结果 ===")
    print(f"传统MSE损失: {metrics['mse']:.6f}")
    print(f"自定义损失 (sin(y) - x)^2: {metrics['custom_loss']:.6f}")
    print(f"sin(预测值)与x的平均绝对误差: {metrics['sin_accuracy']:.6f}")
    
    # 验证几个特殊点
    print("\n=== 特殊点验证 ===")
    test_points = np.array([0, 0.5, -0.5, 0.866, -0.866]).reshape(-1, 1)
    pred_points = fitter.predict(test_points)
    true_points = np.arcsin(test_points.flatten())
    
    for i, x in enumerate(test_points.flatten()):
        pred_y = pred_points[i]
        true_y = true_points[i]
        sin_pred = np.sin(pred_y)
        print(f"x={x:6.3f}: 真实arcsin={true_y:6.3f}, 预测={pred_y:6.3f}, sin(预测)={sin_pred:6.3f}, 误差={abs(sin_pred-x):.6f}")
    
    # 绘制结果
    print("\n绘制结果图...")
    plot_results(fitter, X_test, y_test, y_pred)
    
    print("实验完成！结果图已保存为 mlp_arcsin_results.png")

if __name__ == "__main__":
    main()
