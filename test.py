import numpy as np
from model import NeuralNetwork
from data_loader import load_cifar10
import argparse

def test_model(model_path, test_data, test_labels):
    """
    加载训练好的模型并在测试集上进行评估
    """
    # 初始化模型
    input_size = test_data.shape[1]
    model = NeuralNetwork(input_size, hidden_size=100, output_size=10)
    
    # 加载模型权重
    model.load_weights(model_path)
    
    # 在测试集上进行预测
    test_pred = model.predict(test_data)
    test_acc = np.mean(test_pred == test_labels)
    
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # 打印每个类别的准确率
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(10):
        class_mask = test_labels == i
        class_acc = np.mean(test_pred[class_mask] == test_labels[class_mask])
        print(f'{class_names[i]}: {class_acc:.4f}')

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试CIFAR-10分类器')
    parser.add_argument('--model_path', type=str, default='best_model.npz',
                      help='模型权重文件路径')
    args = parser.parse_args()
    
    # 加载数据
    data_dir = 'cifar-10-batches-py'
    _, (test_data, test_labels) = load_cifar10(data_dir)
    
    # 测试模型
    test_model(args.model_path, test_data, test_labels)

if __name__ == '__main__':
    main() 