import numpy as np
from train import train_model
from data_loader import load_cifar10
import os

def hyperparameter_search():
    """
    执行超参数搜索
    """
    # 加载数据
    data_dir = 'cifar-10-batches-py'
    (train_data, train_labels), (test_data, test_labels) = load_cifar10(data_dir)
    
    # 划分验证集
    num_val = 5000
    val_data = train_data[-num_val:]
    val_labels = train_labels[-num_val:]
    train_data = train_data[:-num_val]
    train_labels = train_labels[:-num_val]
    
    # 定义要搜索的超参数
    hidden_sizes = [50, 100, 200]
    learning_rates = [0.001, 0.01, 0.1]
    reg_lambdas = [0.001, 0.01, 0.1]
    activations = ['relu', 'sigmoid']
    
    # 存储最佳结果
    best_val_acc = 0
    best_params = {}
    
    # 执行网格搜索
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for reg_lambda in reg_lambdas:
                for activation in activations:
                    print(f'\nTesting parameters:')
                    print(f'  Hidden Size: {hidden_size}')
                    print(f'  Learning Rate: {lr}')
                    print(f'  Regularization: {reg_lambda}')
                    print(f'  Activation: {activation}')
                    
                    # 训练模型
                    model = train_model(
                        train_data, train_labels, val_data, val_labels,
                        hidden_size=hidden_size,
                        activation=activation,
                        learning_rate=lr,
                        reg_lambda=reg_lambda,
                        batch_size=128,
                        num_epochs=20,  # 为了加快搜索，使用较少的epochs
                        model_save_path=f'model_{hidden_size}_{lr}_{reg_lambda}_{activation}.npz'
                    )
                    
                    # 在验证集上评估
                    val_pred = model.predict(val_data)
                    val_acc = np.mean(val_pred == val_labels)
                    print(f'  Validation Accuracy: {val_acc:.4f}')
                    
                    # 更新最佳结果
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'reg_lambda': reg_lambda,
                            'activation': activation
                        }
                        print(f'  New best parameters found!')
    
    # 打印最佳参数
    print('\nBest Parameters:')
    for param, value in best_params.items():
        print(f'  {param}: {value}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    
    # 使用最佳参数训练最终模型
    print('\nTraining final model with best parameters...')
    final_model = train_model(
        train_data, train_labels, val_data, val_labels,
        hidden_size=best_params['hidden_size'],
        activation=best_params['activation'],
        learning_rate=best_params['learning_rate'],
        reg_lambda=best_params['reg_lambda'],
        batch_size=128,
        num_epochs=50,
        model_save_path='best_model.npz'
    )
    
    # 在测试集上评估
    test_pred = final_model.predict(test_data)
    test_acc = np.mean(test_pred == test_labels)
    print(f'Final Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    hyperparameter_search() 