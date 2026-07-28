# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
MLP Predictor Training Script
基于 FREEDOM 第一阶段嵌入训练 MLP 二分类预测器
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from time import time

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import init_seed
from models.mlp_predictor import build_mlp_predictor
from utils.binary_evaluator import BinaryEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP Predictor')
    parser.add_argument('-d', '--dataset', type=str, default='tsRNA-disease2',
                        help='Dataset name')
    parser.add_argument('--embeddings', type=str, default=None,
                        help='Path to FREEDOM embeddings file (.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--stopping_step', type=int, default=10,
                        help='Early stopping steps')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    return parser.parse_args()


def load_embeddings(embeddings_path, device):
    """加载 FREEDOM 保存的嵌入"""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    data = torch.load(embeddings_path, map_location='cpu')
    
    user_embeddings = data['user_embeddings'].to(device)
    item_embeddings = data['item_embeddings'].to(device)
    n_users = data['n_users']
    n_items = data['n_items']
    embedding_dim = data['embedding_dim']
    
    print(f"Loaded embeddings from {embeddings_path}")
    print(f"  - Users: {n_users}, Items: {n_items}")
    print(f"  - Embedding dim: {embedding_dim}")
    
    return user_embeddings, item_embeddings, n_users, n_items, embedding_dim


def create_data_loaders(config, train_dataset, valid_dataset, test_dataset, batch_size, device):
    """创建数据加载器"""
    from utils.dataloader import TrainDataLoader, EvalDataLoader
    
    train_data = TrainDataLoader(config, train_dataset, batch_size=batch_size, shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=batch_size)
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=batch_size)
    
    return train_data, valid_data, test_data


class MLPTrainer:
    """MLP 训练器"""
    
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        
        self.epochs = config['mlp_epochs'] if 'mlp_epochs' in config else 100
        self.batch_size = config['mlp_batch_size'] if 'mlp_batch_size' in config else 256
        self.learning_rate = config['mlp_lr'] if 'mlp_lr' in config else 0.001
        self.stopping_step = config['stopping_step'] if 'stopping_step' in config else 10
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        self.best_valid_score = -1
        self.best_valid_result = None
        self.best_test_upon_valid = None
        self.cur_step = 0
        
        self.binary_evaluator = BinaryEvaluator(config)
    
    def _train_epoch(self, train_data, train_dataset):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        # 遍历训练数据
        df = train_dataset.df
        n_samples = len(df)
        
        # 打乱数据
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # 获取批次数据
            user_ids = torch.tensor(df[uid_field].iloc[batch_indices].values, dtype=torch.long).to(self.device)
            item_ids = torch.tensor(df[iid_field].iloc[batch_indices].values, dtype=torch.long).to(self.device)
            labels = torch.ones(len(batch_indices), dtype=torch.float).to(self.device)  # 正样本标签为1
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 添加负样本训练
        neg_loss = self._train_negative_samples(train_data, train_dataset)
        total_loss += neg_loss
        
        return total_loss / (num_batches + 1)
    
    def _train_negative_samples(self, train_data, train_dataset):
        """训练负样本"""
        self.model.train()
        
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        # 获取用户历史交互
        history_items = train_data.history_items_per_u if hasattr(train_data, 'history_items_per_u') else {}
        
        # 获取所有物品
        all_items = set(train_dataset.df[iid_field].unique())
        n_items = len(all_items)
        
        # 采样负样本
        df = train_dataset.df
        n_samples = min(len(df), self.batch_size * 10)  # 限制负样本数量
        
        neg_users = []
        neg_items = []
        
        for user_id in df[uid_field].unique()[:n_samples]:
            user_history = history_items.get(user_id, set())
            # 随机选择一个不在历史中的物品作为负样本
            neg_item = np.random.choice(list(all_items - user_history)) if len(all_items - user_history) > 0 else np.random.choice(list(all_items))
            neg_users.append(user_id)
            neg_items.append(neg_item)
        
        if len(neg_users) == 0:
            return 0.0
        
        user_ids = torch.tensor(neg_users, dtype=torch.long).to(self.device)
        item_ids = torch.tensor(neg_items, dtype=torch.long).to(self.device)
        labels = torch.zeros(len(neg_users), dtype=torch.float).to(self.device)  # 负样本标签为0
        
        self.optimizer.zero_grad()
        predictions = self.model(user_ids, item_ids)
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, eval_data, train_dataset):
        """评估模型"""
        self.model.eval()
        
        n_users = train_dataset.get_user_num()
        n_items = train_dataset.get_item_num()
        
        # 构建 ground truth
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        ground_truth_df = eval_data.dataset.df[[uid_field, iid_field]].copy()
        ground_truth_df.columns = ['userID', 'itemID']
        
        # 获取所有预测分数
        all_scores = []
        with torch.no_grad():
            for user_id in range(n_users):
                scores = self.model.predict_all_items(user_id, self.device)
                all_scores.append(scores.cpu())
        
        scores_matrix = torch.stack(all_scores, dim=0)  # [n_users, n_items]
        
        # 构建标签矩阵
        labels_matrix = torch.zeros(n_users, n_items)
        for _, row in ground_truth_df.iterrows():
            user_id = int(row['userID'])
            item_id = int(row['itemID'])
            if 0 <= user_id < n_users and 0 <= item_id < n_items:
                labels_matrix[user_id, item_id] = 1.0
        
        # 评估
        return self.binary_evaluator.evaluate(scores_matrix, labels_matrix)
    
    def fit(self, train_data, valid_data, test_data, train_dataset):
        """训练模型"""
        print("\n" + "="*50)
        print("Starting MLP Training")
        print("="*50)
        
        for epoch in range(self.epochs):
            start_time = time()
            
            # 训练
            train_loss = self._train_epoch(train_data, train_dataset)
            
            # 评估
            valid_result = self.evaluate(valid_data, train_dataset)
            test_result = self.evaluate(test_data, train_dataset)
            
            valid_score = valid_result.get('AUC', 0.0)
            
            end_time = time()
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.epochs} [time: {end_time-start_time:.2f}s, loss: {train_loss:.4f}]")
            print(f"  Valid: AUC={valid_result['AUC']:.4f}, AUPR={valid_result['AUPR']:.4f}, "
                  f"Precision={valid_result['Precision']:.4f}, Recall={valid_result['Recall']:.4f}, F1={valid_result['F1']:.4f}")
            print(f"  Test:  AUC={test_result['AUC']:.4f}, AUPR={test_result['AUPR']:.4f}, "
                  f"Precision={test_result['Precision']:.4f}, Recall={test_result['Recall']:.4f}, F1={test_result['F1']:.4f}")
            
            # 早停检查
            if valid_score > self.best_valid_score:
                self.best_valid_score = valid_score
                self.best_valid_result = valid_result
                self.best_test_upon_valid = test_result
                self.cur_step = 0
                print("  ✓ Best validation result updated!")
            else:
                self.cur_step += 1
                if self.cur_step >= self.stopping_step:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("\n" + "="*50)
        print("Training Completed")
        print("="*50)
        print(f"Best Valid AUC: {self.best_valid_score:.4f}")
        print(f"Best Valid Result: {self.best_valid_result}")
        print(f"Best Test Result: {self.best_test_upon_valid}")
        
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    init_seed(args.seed)
    
    # 加载配置
    config = Config('FREEDOM', args.dataset, {})
    config['device'] = device
    config['mlp_epochs'] = args.epochs
    config['mlp_batch_size'] = args.batch_size
    config['mlp_lr'] = args.lr
    config['mlp_dropout'] = args.dropout
    config['stopping_step'] = args.stopping_step
    config['seed'] = args.seed
    
    # 初始化日志
    init_logger(config)
    
    # 加载数据集
    dataset = RecDataset(config)
    print(dataset)
    
    train_dataset, valid_dataset, test_dataset = dataset.split()
    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Valid: {len(valid_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # 确定嵌入文件路径
    if args.embeddings is None:
        embeddings_path = os.path.join(config['data_path'], args.dataset, 'freedom_embeddings.pt')
    else:
        embeddings_path = args.embeddings
    
    # 加载嵌入
    user_embeddings, item_embeddings, n_users, n_items, embedding_dim = load_embeddings(
        embeddings_path, device
    )
    
    # 创建数据加载器
    train_data, valid_data, test_data = create_data_loaders(
        config, train_dataset, valid_dataset, test_dataset, args.batch_size, device
    )
    
    # 构建模型
    model = build_mlp_predictor(config, user_embeddings, item_embeddings)
    model = model.to(device)
    print(f"\nModel Architecture:\n{model}")
    
    # 训练
    trainer = MLPTrainer(config, model, device)
    best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
        train_data, valid_data, test_data, train_dataset
    )
    
    return best_valid_score, best_valid_result, best_test_upon_valid


if __name__ == '__main__':
    main()