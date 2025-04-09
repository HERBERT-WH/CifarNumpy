import numpy as np
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import queue

def load_cifar10(data_dir='cifar-10-batches-py'):
    """
    加载CIFAR-10数据集
    :param data_dir: 数据目录
    :return: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # 加载训练数据
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch = unpickle(batch_file)
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    X_train = np.vstack(train_data)
    y_train = np.array(train_labels)
    
    # 加载测试数据
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # 划分验证集
    num_val = 5000
    X_val = X_train[-num_val:]
    y_val = y_train[-num_val:]
    X_train = X_train[:-num_val]
    y_train = y_train[:-num_val]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class DataLoader:
    def __init__(self, data_dir, batch_size=128, num_workers=4, prefetch_factor=2):
        """
        初始化数据加载器
        :param data_dir: 数据目录
        :param batch_size: 批量大小
        :param num_workers: 数据预取线程数
        :param prefetch_factor: 预取因子
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # 加载数据
        self.train_data, self.train_labels, self.test_data, self.test_labels = self._load_data()
        
        # 数据队列
        self.data_queue = queue.Queue(maxsize=prefetch_factor)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # 开始预取数据
        self._start_prefetch()
    
    def _load_cifar10_batch(self, file_path):
        """加载单个CIFAR-10批次文件"""
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        return batch
    
    def _load_data(self):
        """加载完整的CIFAR-10数据集"""
        # 加载训练数据
        train_data = []
        train_labels = []
        for i in range(1, 6):
            batch_file = os.path.join(self.data_dir, f'data_batch_{i}')
            batch = self._load_cifar10_batch(batch_file)
            train_data.append(batch[b'data'])
            train_labels.append(batch[b'labels'])
        
        # 加载测试数据
        test_batch = self._load_cifar10_batch(os.path.join(self.data_dir, 'test_batch'))
        test_data = test_batch[b'data']
        test_labels = test_batch[b'labels']
        
        # 合并训练数据
        train_data = np.vstack(train_data)
        train_labels = np.hstack(train_labels)
        
        # 数据预处理
        train_data = train_data.astype(np.float32) / 255.0
        test_data = test_data.astype(np.float32) / 255.0
        
        return train_data, train_labels, test_data, test_labels
    
    def _prefetch_batch(self):
        """预取一个批次的数据"""
        while True:
            try:
                indices = np.random.choice(len(self.train_data), self.batch_size, replace=False)
                batch_X = self.train_data[indices]
                batch_y = self.train_labels[indices]
                self.data_queue.put((batch_X, batch_y))
            except Exception as e:
                print(f"预取数据时出错: {e}")
                break
    
    def _start_prefetch(self):
        """启动数据预取"""
        for _ in range(self.num_workers):
            self.executor.submit(self._prefetch_batch)
    
    def get_batch(self):
        """获取一个批次的数据"""
        try:
            return self.data_queue.get(timeout=1)
        except queue.Empty:
            print("警告: 数据队列为空，重新启动预取")
            self._start_prefetch()
            return self.get_batch()
    
    def get_test_data(self):
        """获取测试数据"""
        return self.test_data, self.test_labels
    
    def __del__(self):
        """清理资源"""
        self.executor.shutdown(wait=False) 