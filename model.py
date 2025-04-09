import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        初始化三层神经网络
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        :param activation: 激活函数类型，支持 'relu' 和 'sigmoid'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # 用于存储训练历史
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def _activation(self, x, derivative=False):
        """激活函数"""
        if self.activation == 'relu':
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            if derivative:
                sig = 1 / (1 + np.exp(-x))
                return sig * (1 - sig)
            return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        """前向传播"""
        # 第一层
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activation(self.Z1)
        
        # 第二层（输出层）
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # 输出层使用softmax
        exp_scores = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return self.A2
    
    def backward(self, X, y, learning_rate, reg_lambda):
        """反向传播"""
        m = X.shape[0]
        
        # 计算输出层误差
        dZ2 = self.A2
        dZ2[range(m), y] -= 1
        dZ2 /= m
        
        # 计算第二层梯度
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 计算隐藏层误差
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activation(self.Z1, derivative=True)
        
        # 计算第一层梯度
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # 添加L2正则化
        dW2 += reg_lambda * self.W2
        dW1 += reg_lambda * self.W1
        
        # 更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def predict(self, X):
        """预测"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, X, y, reg_lambda):
        """计算损失函数"""
        m = X.shape[0]
        probs = self.forward(X)
        
        # 计算交叉熵损失
        correct_logprobs = -np.log(probs[range(m), y])
        data_loss = np.sum(correct_logprobs) / m
        
        # 添加L2正则化
        reg_loss = 0.5 * reg_lambda * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        
        return data_loss + reg_loss
    
    def train(self, X, y, X_val, y_val, learning_rate=0.01, reg_lambda=0.01, 
              epochs=100, batch_size=32, verbose=True):
        """训练模型"""
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_batches):
                # 获取小批量数据
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播和反向传播
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, reg_lambda)
            
            # 计算训练集和验证集的损失和准确率
            train_loss = self.compute_loss(X, y, reg_lambda)
            train_pred = self.predict(X)
            train_acc = np.mean(train_pred == y)
            
            val_loss = self.compute_loss(X_val, y_val, reg_lambda)
            val_pred = self.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            
            # 记录训练历史
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("-" * 50)
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['accuracy'], label='Train Acc')
        plt.plot(self.training_history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def plot_confusion_matrix(self, X, y, class_names):
        """绘制混淆矩阵"""
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def plot_weights(self):
        """绘制权重分布"""
        plt.figure(figsize=(12, 4))
        
        # 绘制第一层权重分布
        plt.subplot(1, 2, 1)
        plt.hist(self.W1.flatten(), bins=50)
        plt.title('First Layer Weights Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
        # 绘制第二层权重分布
        plt.subplot(1, 2, 2)
        plt.hist(self.W2.flatten(), bins=50)
        plt.title('Second Layer Weights Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('weights_distribution.png')
        plt.close()
    
    def save_weights(self, filepath):
        """保存模型权重"""
        weights = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'activation': self.activation,
            'training_history': self.training_history
        }
        np.savez(filepath, **weights)
    
    def load_weights(self, filepath):
        """加载模型权重"""
        weights = np.load(filepath, allow_pickle=True)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        self.input_size = weights['input_size']
        self.hidden_size = weights['hidden_size']
        self.output_size = weights['output_size']
        self.activation = weights['activation'].item()
        self.training_history = weights['training_history'].item()