import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

class ArcsinMLP(nn.Module):
    """
    MLP network implemented with PyTorch for fitting arcsin function
    """
    def __init__(self, hidden_sizes=[128, 64, 32]):
        super(ArcsinMLP, self).__init__()
        
        layers = []
        input_size = 1
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class ArcsinTrainer:
    """
    Trainer class implementing custom loss function
    """
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.custom_losses = []
        self.mse_losses = []
    
    def custom_loss_function(self, y_pred, x):
        """
        Custom loss function: loss = (sin(y_pred) - x)^2
        """
        sin_y_pred = torch.sin(y_pred)
        loss = torch.mean((sin_y_pred - x) ** 2)
        return loss
    
    def combined_loss_function(self, y_pred, y_true, x, alpha=0.7):
        """
        Combined loss function: alpha * custom_loss + (1-alpha) * mse_loss
        """
        custom_loss = self.custom_loss_function(y_pred, x)
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        
        total_loss = alpha * custom_loss + (1 - alpha) * mse_loss
        return total_loss, custom_loss, mse_loss
    
    def train_epoch(self, dataloader):
        """Train one epoch using only custom loss"""
        self.model.train()
        total_loss = 0
        total_custom_loss = 0
        total_mse_loss = 0
        
        for batch_x, batch_y_true in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y_true = batch_y_true.to(self.device)
            
            y_pred = self.model(batch_x)
            
            loss = self.custom_loss_function(y_pred, batch_x)
            custom_loss = loss
            mse_loss = nn.functional.mse_loss(y_pred, batch_y_true)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_custom_loss += custom_loss.item()
            total_mse_loss += mse_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_custom_loss = total_custom_loss / len(dataloader)
        avg_mse_loss = total_mse_loss / len(dataloader)
        
        return avg_loss, avg_custom_loss, avg_mse_loss
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_custom_loss = 0
        total_mse_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y_true in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y_true = batch_y_true.to(self.device)
                
                y_pred = self.model(batch_x)
                
                custom_loss = self.custom_loss_function(y_pred, batch_x)
                mse_loss = nn.functional.mse_loss(y_pred, batch_y_true)
                
                total_custom_loss += custom_loss.item()
                total_mse_loss += mse_loss.item()
        
        avg_custom_loss = total_custom_loss / len(dataloader)
        avg_mse_loss = total_mse_loss / len(dataloader)
        
        return avg_custom_loss, avg_mse_loss
    
    def train(self, train_loader, val_loader, epochs=200):
        """
        Train model using only custom loss function
        """
        print("Starting training (custom loss only)...")
        
        best_custom_loss = float('inf')
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(epochs):
            train_loss, train_custom_loss, train_mse_loss = self.train_epoch(train_loader)
            
            val_custom_loss, val_mse_loss = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.custom_losses.append(val_custom_loss)
            self.mse_losses.append(val_mse_loss)
            
            self.scheduler.step(val_custom_loss)
            
            if val_custom_loss < best_custom_loss:
                best_custom_loss = val_custom_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_arcsin_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Custom Loss={val_custom_loss:.6f}, MSE={val_mse_loss:.6f}")
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('best_arcsin_model.pth'))
        print("Training completed!")
    
    def predict(self, x):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            return self.model(x).cpu().numpy()

def generate_data(n_samples=2000, noise_level=0.0):
    """Generate training data"""
    x = np.random.uniform(-0.99, 0.99, n_samples)
    y_true = np.arcsin(x)
    
    if noise_level > 0:
        y_true += np.random.normal(0, noise_level, n_samples)
    
    return x.reshape(-1, 1), y_true.reshape(-1, 1)

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=64):
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

def plot_results(model, trainer, X_test, y_test):
    """Plot training results"""
    y_pred = trainer.predict(X_test)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].scatter(X_test.flatten(), y_test.flatten(), alpha=0.6, label='True arcsin(x)', color='blue', s=10)
    axes[0, 0].scatter(X_test.flatten(), y_pred.flatten(), alpha=0.6, label='MLP Prediction', color='red', s=10)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('arcsin(x)')
    axes[0, 0].set_title('MLP Prediction vs True arcsin Function')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    sin_y_pred = np.sin(y_pred)
    axes[0, 1].scatter(X_test.flatten(), X_test.flatten(), alpha=0.6, label='y=x (Ideal)', color='blue', s=10)
    axes[0, 1].scatter(X_test.flatten(), sin_y_pred.flatten(), alpha=0.6, label='sin(MLP Prediction)', color='red', s=10)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('sin(Prediction)')
    axes[0, 1].set_title('sin(Prediction) vs x (Custom Loss Verification)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    x_continuous = np.linspace(-0.99, 0.99, 300).reshape(-1, 1)
    y_true_continuous = np.arcsin(x_continuous.flatten())
    y_pred_continuous = trainer.predict(x_continuous)
    
    axes[0, 2].plot(x_continuous.flatten(), y_true_continuous, label='True arcsin(x)', color='blue', linewidth=2)
    axes[0, 2].plot(x_continuous.flatten(), y_pred_continuous.flatten(), label='MLP Fit', color='red', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('arcsin(x)')
    axes[0, 2].set_title('Continuous Function Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    epochs = range(len(trainer.custom_losses))
    axes[1, 0].plot(epochs, trainer.custom_losses, label='Custom Loss', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Custom Loss')
    axes[1, 0].set_title('Training Custom Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, trainer.mse_losses, label='MSE Loss', color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_title('Training MSE Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    error = y_pred.flatten() - y_test.flatten()
    axes[1, 2].hist(error, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_xlabel('Prediction Error')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Error Distribution')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/torch_arcsin_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred

def evaluate_model(trainer, X_test, y_test, y_pred):
    """Evaluate model performance"""
    mse = np.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    
    sin_y_pred = np.sin(y_pred.flatten())
    custom_loss = np.mean((sin_y_pred - X_test.flatten()) ** 2)
    
    sin_accuracy = np.mean(np.abs(sin_y_pred - X_test.flatten()))
    
    ss_res = np.sum((y_test.flatten() - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_test.flatten() - np.mean(y_test.flatten())) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'custom_loss': custom_loss,
        'sin_accuracy': sin_accuracy,
        'r2_score': r2_score
    }

def main():
    """Main function"""
    print("=== PyTorch MLP Arcsin Function Fitting Experiment ===")
    print("Objective: Train MLP to minimize loss = (sin(MLP(x)) - x)²")
    print("Note: Using custom loss function only")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Generating training data...")
    X_train, y_train = generate_data(n_samples=3000)
    X_val, y_val = generate_data(n_samples=800)
    X_test, y_test = generate_data(n_samples=500)
    
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=128)
    
    model = ArcsinMLP(hidden_sizes=[256, 128, 64, 32])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = ArcsinTrainer(model, learning_rate=0.001, device=device)
    
    trainer.train(train_loader, val_loader, epochs=300)
    
    print("\nPlotting results...")
    y_pred = plot_results(model, trainer, X_test, y_test)
    
    print("Evaluating model performance...")
    metrics = evaluate_model(trainer, X_test, y_test, y_pred)
    
    print("\n=== Evaluation Results ===")
    print(f"MSE Loss: {metrics['mse']:.6f}")
    print(f"Custom Loss (sin(y) - x)²: {metrics['custom_loss']:.6f}")
    print(f"sin(prediction) vs x MAE: {metrics['sin_accuracy']:.6f}")
    print(f"R² Score: {metrics['r2_score']:.6f}")
    
    print("\n=== Special Points Verification ===")
    test_points = np.array([0, 0.5, -0.5, 0.866, -0.866, 0.25, -0.25]).reshape(-1, 1)
    pred_points = trainer.predict(test_points)
    true_points = np.arcsin(test_points.flatten())
    
    for i, x in enumerate(test_points.flatten()):
        pred_y = pred_points[i, 0]
        true_y = true_points[i]
        sin_pred = np.sin(pred_y)
        error = abs(sin_pred - x)
        print(f"x={x:6.3f}: true arcsin={true_y:6.3f}, pred={pred_y:6.3f}, "
              f"sin(pred)={sin_pred:6.3f}, error={error:.6f}")
    
    print("\nExperiment completed! Results saved as torch_arcsin_results.png")

if __name__ == "__main__":
    main()
