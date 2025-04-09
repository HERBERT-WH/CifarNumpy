import numpy as np
from model import NeuralNetwork
from data_loader import DataLoader, load_cifar10
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # 保存数据到CSV文件
    df = pd.DataFrame(history)
    df.to_csv('training_history.csv', index=False)

class LearningRateScheduler:
    def __init__(self, initial_lr, decay_rate=0.95, decay_steps=5):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_lr = initial_lr
        self.step = 0
    
    def get_lr(self):
        """获取当前学习率"""
        if self.step % self.decay_steps == 0 and self.step > 0:
            self.current_lr *= self.decay_rate
        self.step += 1
        return self.current_lr

def train_model(train_data, train_labels, val_data, val_labels, 
                hidden_size=100, activation='relu',
                learning_rate=0.01, reg_lambda=0.01,
                batch_size=128, num_epochs=50,
                model_save_path='best_model.npz',
                checkpoint_dir='checkpoints',
                checkpoint_freq=5,
                num_workers=4,
                prefetch_factor=2):
    """
    训练神经网络模型
    :param train_data: 训练数据
    :param train_labels: 训练标签
    :param val_data: 验证数据
    :param val_labels: 验证标签
    :param hidden_size: 隐藏层大小
    :param activation: 激活函数类型
    :param learning_rate: 学习率
    :param reg_lambda: 正则化强度
    :param batch_size: 批量大小
    :param num_epochs: 训练轮数
    :param model_save_path: 最佳模型保存路径
    :param checkpoint_dir: checkpoint保存目录
    :param checkpoint_freq: checkpoint保存频率（每隔多少轮保存一次）
    :param num_workers: 数据预取线程数
    :param prefetch_factor: 预取因子
    """
    input_size = train_data.shape[1]
    output_size = 10  # CIFAR-10有10个类别
    
    # 创建checkpoint目录
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 初始化模型
    model = NeuralNetwork(input_size, hidden_size, output_size, activation)
    
    # 初始化数据加载器
    data_loader = DataLoader('cifar-10-batches-py', batch_size, num_workers, prefetch_factor)
    
    # 初始化学习率调度器
    lr_scheduler = LearningRateScheduler(learning_rate)
    
    # 训练参数
    best_val_acc = 0
    num_train = train_data.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    
    # 记录训练历史
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'time_per_epoch': []
    }
    
    print("\n开始训练...")
    print(f"训练集大小: {num_train}")
    print(f"批量大小: {batch_size}")
    print(f"每个epoch的迭代次数: {iterations_per_epoch}")
    print(f"总训练轮数: {num_epochs}")
    print(f"数据预取线程数: {num_workers}")
    print("-" * 50)
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_start_time = time()
        epoch_loss = 0
        
        # 每个epoch的迭代
        for i in range(iterations_per_epoch):
            # 获取当前学习率
            current_lr = lr_scheduler.get_lr()
            
            # 获取小批量数据
            batch_X, batch_y = data_loader.get_batch()
            
            # 前向传播和反向传播
            model.forward(batch_X)
            model.backward(batch_X, batch_y, current_lr, reg_lambda)
            
            # 计算当前批次的损失
            batch_loss = model.compute_loss(batch_X, batch_y, reg_lambda)
            epoch_loss += batch_loss
            
            # 显示进度
            if (i + 1) % 10 == 0 or i == iterations_per_epoch - 1:
                progress = (i + 1) / iterations_per_epoch * 100
                avg_loss = epoch_loss / (i + 1)
                print(f"\rEpoch {epoch + 1}/{num_epochs} - {progress:.1f}% - 平均损失: {avg_loss:.4f} - 学习率: {current_lr:.6f}", end="")
        
        # 计算训练集和验证集损失和准确率
        train_loss = model.compute_loss(train_data, train_labels, reg_lambda)
        train_pred = model.predict(train_data)
        train_acc = np.mean(train_pred == train_labels)
        
        val_loss = model.compute_loss(val_data, val_labels, reg_lambda)
        val_pred = model.predict(val_data)
        val_acc = np.mean(val_pred == val_labels)
        
        # 计算epoch耗时
        epoch_time = time() - epoch_start_time
        
        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} 完成 (耗时: {epoch_time:.2f}秒):")
        print(f"  训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"  验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(model_save_path)
            print(f"  新的最佳模型已保存! 验证集准确率: {val_acc:.4f}")
        
        # 保存checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.npz')
            model.save_weights(checkpoint_path)
            print(f"  Checkpoint已保存 (Epoch {epoch+1})")
        
        print("-" * 50)
    
    # 绘制训练历史图表
    plot_training_history(history)
    
    return model

def main():
    # 加载数据
    print("正在加载数据...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10()
    
    # 数据预处理
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    # 创建模型
    input_size = X_train.shape[1]
    hidden_size = 512
    output_size = 10
    model = NeuralNetwork(input_size, hidden_size, output_size, activation='relu')
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.01
    reg_lambda = 0.01
    epochs = 100
    
    # 计算批次数量
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size
    
    print(f"\n训练参数:")
    print(f"训练集大小: {n_train}")
    print(f"批量大小: {batch_size}")
    print(f"每轮迭代次数: {n_batches}")
    print(f"总训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"正则化强度: {reg_lambda}")
    print("-" * 50)
    
    # 训练模型
    print("\n开始训练...")
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # 初始化每轮的损失和准确率
        epoch_loss = 0
        epoch_correct = 0
        
        # 使用tqdm显示进度条
        with tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for i in pbar:
                # 获取当前批次
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_train)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # 前向传播和反向传播
                model.forward(X_batch)
                model.backward(X_batch, y_batch, learning_rate, reg_lambda)
                
                # 计算当前批次的损失和准确率
                batch_loss = model.compute_loss(X_batch, y_batch, reg_lambda)
                batch_pred = model.predict(X_batch)
                batch_acc = np.mean(batch_pred == y_batch)
                
                # 更新累计值
                epoch_loss += batch_loss
                epoch_correct += np.sum(batch_pred == y_batch)
                
                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_correct / n_train
        
        # 在验证集上评估
        val_pred = model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        val_loss = model.compute_loss(X_val, y_val, reg_lambda)
        
        # 打印每轮的结果
        print(f"\nEpoch {epoch+1}/{epochs} 完成:")
        print(f"  训练集 - 损失: {avg_loss:.4f}, 准确率: {avg_acc:.4f}")
        print(f"  验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        print("-" * 50)
    
    # 评估模型
    print("\n评估模型...")
    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 可视化
    print("\n生成可视化图表...")
    # 1. 训练历史
    model.plot_training_history()
    
    # 2. 混淆矩阵
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    model.plot_confusion_matrix(X_test, y_test, class_names)
    
    # 3. 权重分布
    model.plot_weights()
    
    # 保存模型
    print("\n保存模型...")
    model.save_weights('best_model.npz')
    print("模型已保存到 best_model.npz")

if __name__ == "__main__":
    main() 