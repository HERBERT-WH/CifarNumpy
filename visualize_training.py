import pandas as pd
import matplotlib.pyplot as plt

def visualize_training_history(csv_path):
    """
    可视化训练历史数据，包括训练损失、验证损失、训练准确率和验证准确率
    
    参数:
        csv_path (str): 训练历史CSV文件的路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy ')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    visualize_training_history('training_history.csv') 