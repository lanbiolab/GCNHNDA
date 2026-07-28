# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
5-Fold Cross Validation for FREEDOM + MLP
完整两阶段交叉验证
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
from time import time
from logging import getLogger

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import init_seed, get_model, get_trainer, dict2str
from utils.dataloader import TrainDataLoader, EvalDataLoader
from models.mlp_predictor import build_mlp_predictor
from utils.binary_evaluator import BinaryEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='5-Fold Cross Validation for FREEDOM + MLP')
    parser.add_argument('-d', '--dataset', type=str, default='tsRNA-disease2',
                        help='Dataset name')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for K-fold split')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio from training set')
    # FREEDOM 参数
    parser.add_argument('--freedom_epochs', type=int, default=300,
                        help='FREEDOM training epochs')
    parser.add_argument('--freedom_lr', type=float, default=0.001,
                        help='FREEDOM learning rate')
    parser.add_argument('--stopping_step', type=int, default=20,
                        help='Early stopping steps for FREEDOM')
    # MLP 参数
    parser.add_argument('--mlp_epochs', type=int, default=100,
                        help='MLP training epochs')
    parser.add_argument('--mlp_lr', type=float, default=0.001,
                        help='MLP learning rate')
    parser.add_argument('--mlp_batch_size', type=int, default=256,
                        help='MLP batch size')
    parser.add_argument('--mlp_dropout', type=float, default=0.3,
                        help='MLP dropout rate')
    parser.add_argument('--mlp_stopping_step', type=int, default=10,
                        help='Early stopping steps for MLP')
    # 其他
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    return parser.parse_args()


def train_freedom_fold(config, train_df, val_df, test_df, fold_idx, device):
    """
    训练单个 fold 的 FREEDOM 模型
    
    Returns:
        model: 训练好的模型
        results: 评估结果
    """
    logger = getLogger()
    
    # 创建数据集
    full_dataset = RecDataset(config)
    train_dataset = full_dataset.copy(train_df)
    
    # 过滤验证集和测试集中训练集没有的用户
    uid_field = config['USER_ID_FIELD']
    train_users = set(train_df[uid_field].unique())
    
    val_df_filtered = val_df[val_df[uid_field].isin(train_users)].copy()
    test_df_filtered = test_df[test_df[uid_field].isin(train_users)].copy()
    
    val_dataset = full_dataset.copy(val_df_filtered)
    test_dataset = full_dataset.copy(test_df_filtered)
    
    logger.info(f'\n=== Fold {fold_idx+1} FREEDOM Training ===')
    logger.info(f'Train: {len(train_dataset)}, Val: {len(val_dataset)} (filtered from {len(val_df)}), Test: {len(test_dataset)} (filtered from {len(test_df)})')
    
    # 创建数据加载器
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_data = EvalDataLoader(config, val_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    
    # 初始化模型
    model = get_model(config['model'])(config, train_data).to(device)
    logger.info(f'FREEDOM model initialized')
    
    # 训练
    trainer = get_trainer()(config, model)
    best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
        train_data, valid_data=val_data, test_data=test_data, saved=False
    )
    
    # 保存嵌入
    embeddings_path = os.path.join(config['data_path'], config['dataset'], f'freedom_embeddings_fold{fold_idx}.pt')
    model.save_embeddings(embeddings_path)
    
    # 获取测试集曲线数据（使用 BinaryEvaluator）
    binary_evaluator = BinaryEvaluator(config)
    n_users = train_dataset.get_user_num()
    n_items = train_dataset.get_item_num()
    
    # 构建测试集标签矩阵
    test_labels_matrix = torch.zeros(n_users, n_items)
    for _, row in test_df_filtered.iterrows():
        user_id = int(row[uid_field])
        item_id = int(row[config['ITEM_ID_FIELD']])
        if 0 <= user_id < n_users and 0 <= item_id < n_items:
            test_labels_matrix[user_id, item_id] = 1.0
    
    # 获取预测分数矩阵
    model.eval()
    with torch.no_grad():
        all_scores = []
        for user_id in range(n_users):
            scores = model.full_sort_predict([torch.tensor([user_id]).to(device)])
            all_scores.append(scores.cpu())
        scores_matrix = torch.cat(all_scores, dim=0)
    
    # 获取曲线数据
    test_curve_data = binary_evaluator.get_curve_data_gmnn_style(scores_matrix, test_labels_matrix)
    
    return model, best_valid_result, best_test_upon_valid, embeddings_path, test_curve_data


def train_mlp_fold(config, embeddings_path, train_df, val_df, test_df, full_dataset, fold_idx, device):
    """
    训练单个 fold 的 MLP 预测器
    
    Returns:
        results: 评估结果
    """
    logger = getLogger()
    
    logger.info(f'\n=== Fold {fold_idx+1} MLP Training ===')
    
    # 加载嵌入
    data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    user_embeddings = data['user_embeddings'].to(device)
    item_embeddings = data['item_embeddings'].to(device)
    
    # 构建模型
    model = build_mlp_predictor(config, user_embeddings, item_embeddings)
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = full_dataset.copy(train_df)
    
    # 过滤验证集和测试集中训练集没有的用户
    uid_field = config['USER_ID_FIELD']
    train_users = set(train_df[uid_field].unique())
    
    val_df_filtered = val_df[val_df[uid_field].isin(train_users)].copy()
    test_df_filtered = test_df[test_df[uid_field].isin(train_users)].copy()
    
    val_dataset = full_dataset.copy(val_df_filtered)
    test_dataset = full_dataset.copy(test_df_filtered)
    
    logger.info(f'MLP Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['mlp_batch_size'], shuffle=True)
    val_data = EvalDataLoader(config, val_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    
    # 训练 MLP
    mlp_trainer = MLPTrainerCV(config, model, device, train_data, train_dataset)
    best_valid_result, best_test_upon_valid = mlp_trainer.fit(val_data, test_data)
    
    # 获取测试集曲线数据
    test_curve_data = mlp_trainer.evaluate_curves(test_data)
    
    return best_valid_result, best_test_upon_valid, test_curve_data


class MLPTrainerCV:
    """MLP 训练器（交叉验证版本）"""
    
    def __init__(self, config, model, device, train_data, train_dataset):
        self.config = config
        self.model = model
        self.device = device
        self.train_data = train_data
        self.train_dataset = train_dataset
        
        self.epochs = config['mlp_epochs'] if 'mlp_epochs' in config else 100
        self.batch_size = config['mlp_batch_size'] if 'mlp_batch_size' in config else 256
        self.learning_rate = config['mlp_lr'] if 'mlp_lr' in config else 0.001
        self.stopping_step = config['mlp_stopping_step'] if 'mlp_stopping_step' in config else 10
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        self.best_valid_score = -1
        self.best_valid_result = None
        self.best_test_upon_valid = None
        self.cur_step = 0
        
        self.binary_evaluator = BinaryEvaluator(config)
    
    def _train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        df = self.train_dataset.df
        n_samples = len(df)
        
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            user_ids = torch.tensor(df[uid_field].iloc[batch_indices].values, dtype=torch.long).to(self.device)
            item_ids = torch.tensor(df[iid_field].iloc[batch_indices].values, dtype=torch.long).to(self.device)
            labels = torch.ones(len(batch_indices), dtype=torch.float).to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 负样本训练
        neg_loss = self._train_negative_samples()
        total_loss += neg_loss
        
        return total_loss / (num_batches + 1)
    
    def _train_negative_samples(self):
        """训练负样本"""
        self.model.train()
        
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        history_items = self.train_data.history_items_per_u if hasattr(self.train_data, 'history_items_per_u') else {}
        all_items = set(self.train_dataset.df[iid_field].unique())
        
        df = self.train_dataset.df
        n_samples = min(len(df), self.batch_size * 10)
        
        neg_users = []
        neg_items = []
        
        for user_id in df[uid_field].unique()[:n_samples]:
            user_history = history_items.get(user_id, set())
            neg_item = np.random.choice(list(all_items - user_history)) if len(all_items - user_history) > 0 else np.random.choice(list(all_items))
            neg_users.append(user_id)
            neg_items.append(neg_item)
        
        if len(neg_users) == 0:
            return 0.0
        
        user_ids = torch.tensor(neg_users, dtype=torch.long).to(self.device)
        item_ids = torch.tensor(neg_items, dtype=torch.long).to(self.device)
        labels = torch.zeros(len(neg_users), dtype=torch.float).to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(user_ids, item_ids)
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, eval_data):
        """评估模型"""
        self.model.eval()
        
        n_users = self.train_dataset.get_user_num()
        n_items = self.train_dataset.get_item_num()
        
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        ground_truth_df = eval_data.dataset.df[[uid_field, iid_field]].copy()
        ground_truth_df.columns = ['userID', 'itemID']
        
        all_scores = []
        with torch.no_grad():
            for user_id in range(n_users):
                scores = self.model.predict_all_items(user_id, self.device)
                all_scores.append(scores.cpu())
        
        scores_matrix = torch.stack(all_scores, dim=0)
        
        labels_matrix = torch.zeros(n_users, n_items)
        for _, row in ground_truth_df.iterrows():
            user_id = int(row['userID'])
            item_id = int(row['itemID'])
            if 0 <= user_id < n_users and 0 <= item_id < n_items:
                labels_matrix[user_id, item_id] = 1.0
        
        return self.binary_evaluator.evaluate(scores_matrix, labels_matrix)
    
    def evaluate_curves(self, eval_data):
        """评估模型并返回曲线数据（参考 GMNN2CD.py 方式）"""
        self.model.eval()
        
        n_users = self.train_dataset.get_user_num()
        n_items = self.train_dataset.get_item_num()
        
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        ground_truth_df = eval_data.dataset.df[[uid_field, iid_field]].copy()
        ground_truth_df.columns = ['userID', 'itemID']
        
        all_scores = []
        with torch.no_grad():
            for user_id in range(n_users):
                scores = self.model.predict_all_items(user_id, self.device)
                all_scores.append(scores.cpu())
        
        scores_matrix = torch.stack(all_scores, dim=0)
        
        labels_matrix = torch.zeros(n_users, n_items)
        for _, row in ground_truth_df.iterrows():
            user_id = int(row['userID'])
            item_id = int(row['itemID'])
            if 0 <= user_id < n_users and 0 <= item_id < n_items:
                labels_matrix[user_id, item_id] = 1.0
        
        return self.binary_evaluator.get_curve_data_gmnn_style(scores_matrix, labels_matrix)
    
    def fit(self, val_data, test_data):
        """训练模型"""
        logger = getLogger()
        
        for epoch in range(self.epochs):
            start_time = time()
            
            train_loss = self._train_epoch()
            
            valid_result = self.evaluate(val_data)
            test_result = self.evaluate(test_data)
            
            valid_score = valid_result.get('AUC', 0.0)
            
            end_time = time()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'  MLP Epoch {epoch+1}/{self.epochs} [time: {end_time-start_time:.2f}s, loss: {train_loss:.4f}]')
                logger.info(f'    Valid: AUC={valid_result["AUC"]:.4f}, Test: AUC={test_result["AUC"]:.4f}')
            
            if valid_score > self.best_valid_score:
                self.best_valid_score = valid_score
                self.best_valid_result = valid_result
                self.best_test_upon_valid = test_result
                self.cur_step = 0
            else:
                self.cur_step += 1
                if self.cur_step >= self.stopping_step:
                    logger.info(f'  MLP Early stopping at epoch {epoch+1}')
                    break
        
        logger.info(f'  MLP Best Valid AUC: {self.best_valid_score:.4f}')
        return self.best_valid_result, self.best_test_upon_valid


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置
    config = Config('FREEDOM', args.dataset, {})
    config['device'] = device
    config['mlp_epochs'] = args.mlp_epochs
    config['mlp_lr'] = args.mlp_lr
    config['mlp_batch_size'] = args.mlp_batch_size
    config['mlp_dropout'] = args.mlp_dropout
    config['mlp_stopping_step'] = args.mlp_stopping_step
    config['stopping_step'] = args.stopping_step
    
    # 设置 FREEDOM 模型的固定超参数（避免列表与 float 比较出错）
    config['dropout'] = 0.3  # 固定 dropout 值
    config['reg_weight'] = 1e-03  # 固定正则化权重
    config['use_hard_neg_mixup'] = True  # 启用困难负样本 Mixup
    config['mixup_alpha_range'] = [0.0, 0.5]  # Mixup 混合系数范围
    
    # 初始化日志
    init_logger(config)
    logger = getLogger()
    
    logger.info('='*60)
    logger.info(f'{args.k}-Fold Cross Validation for FREEDOM + MLP')
    logger.info('='*60)
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'K: {args.k}, Seed: {args.seed}, Val Ratio: {args.val_ratio}')
    
    # 加载数据集
    dataset = RecDataset(config)
    logger.info(str(dataset))
    
    # K 折划分
    kfold_splits = dataset.kfold_split(k=args.k, seed=args.seed, val_ratio=args.val_ratio)
    
    # 存储结果
    all_fold_results = []
    
    # 存储曲线数据
    all_freedom_curves = []
    all_mlp_curves = []
    
    # K 折交叉验证
    for fold_idx, (train_df, val_df, test_df) in enumerate(kfold_splits):
        logger.info(f'\n{"="*60}')
        logger.info(f'FOLD {fold_idx + 1}/{args.k}')
        logger.info(f'{"="*60}')
        logger.info(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
        
        # 阶段 1: 训练 FREEDOM
        init_seed(args.seed + fold_idx)
        freedom_model, freedom_valid, freedom_test, embeddings_path, freedom_curve = train_freedom_fold(
            config, train_df, val_df, test_df, fold_idx, device
        )
        all_freedom_curves.append(freedom_curve)
        
        logger.info(f'FREEDOM Results - Valid: AUC={freedom_valid["AUC"]:.4f}, Test: AUC={freedom_test["AUC"]:.4f}')
        
        # 阶段 2: 训练 MLP
        init_seed(args.seed + fold_idx)
        mlp_valid, mlp_test, mlp_curve = train_mlp_fold(
            config, embeddings_path, train_df, val_df, test_df, dataset, fold_idx, device
        )
        all_mlp_curves.append(mlp_curve)
        
        logger.info(f'MLP Results - Valid: AUC={mlp_valid["AUC"]:.4f}, Test: AUC={mlp_test["AUC"]:.4f}')
        
        # 记录结果
        fold_result = {
            'fold': fold_idx + 1,
            'freedom_valid': freedom_valid,
            'freedom_test': freedom_test,
            'mlp_valid': mlp_valid,
            'mlp_test': mlp_test
        }
        all_fold_results.append(fold_result)
        
        # 清理临时嵌入文件
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
    
    # 汇总结果
    logger.info('\n' + '='*60)
    logger.info('CROSS VALIDATION RESULTS SUMMARY')
    logger.info('='*60)
    
    # 计算平均值和标准差
    metrics = ['AUC', 'AUPR', 'Precision', 'Recall', 'F1', 'Accuracy']
    
    logger.info('\n--- FREEDOM Results ---')
    for metric in metrics:
        valid_scores = [r['freedom_valid'][metric] for r in all_fold_results]
        test_scores = [r['freedom_test'][metric] for r in all_fold_results]
        logger.info(f'{metric}: Valid={np.mean(valid_scores):.4f}±{np.std(valid_scores):.4f}, Test={np.mean(test_scores):.4f}±{np.std(test_scores):.4f}')
    
    logger.info('\n--- MLP Results ---')
    for metric in metrics:
        valid_scores = [r['mlp_valid'][metric] for r in all_fold_results]
        test_scores = [r['mlp_test'][metric] for r in all_fold_results]
        logger.info(f'{metric}: Valid={np.mean(valid_scores):.4f}±{np.std(valid_scores):.4f}, Test={np.mean(test_scores):.4f}±{np.std(test_scores):.4f}')
    
    logger.info('\n--- Detailed Fold Results ---')
    for r in all_fold_results:
        logger.info(f"Fold {r['fold']}: FREEDOM Test AUC={r['freedom_test']['AUC']:.4f}, MLP Test AUC={r['mlp_test']['AUC']:.4f}")
    
    # 保存曲线数据到 h5 文件（参考 GMNN2CD.py 方式）
    logger.info('\n--- Saving Curve Data ---')
    output_dir = os.path.join(os.path.dirname(config['data_path']), 'tsRNA_PlotPng')
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = args.dataset
    
    # 转换为数组并求平均（参考 GMNN2CD.py）
    # FREEDOM 曲线数据
    freedom_tpr_arr = np.array([c['tpr'] for c in all_freedom_curves])
    freedom_fpr_arr = np.array([c['fpr'] for c in all_freedom_curves])
    freedom_precision_arr = np.array([c['precision'] for c in all_freedom_curves])
    freedom_recall_arr = np.array([c['recall'] for c in all_freedom_curves])
    
    mean_freedom_tpr = np.mean(freedom_tpr_arr, axis=0)
    mean_freedom_fpr = np.mean(freedom_fpr_arr, axis=0)
    mean_freedom_precision = np.mean(freedom_precision_arr, axis=0)
    mean_freedom_recall = np.mean(freedom_recall_arr, axis=0)
    
    # MLP 曲线数据
    mlp_tpr_arr = np.array([c['tpr'] for c in all_mlp_curves])
    mlp_fpr_arr = np.array([c['fpr'] for c in all_mlp_curves])
    mlp_precision_arr = np.array([c['precision'] for c in all_mlp_curves])
    mlp_recall_arr = np.array([c['recall'] for c in all_mlp_curves])
    
    mean_mlp_tpr = np.mean(mlp_tpr_arr, axis=0)
    mean_mlp_fpr = np.mean(mlp_fpr_arr, axis=0)
    mean_mlp_precision = np.mean(mlp_precision_arr, axis=0)
    mean_mlp_recall = np.mean(mlp_recall_arr, axis=0)
    
    # 计算真实 AUC 和 AUPR 值（来自各 fold 的评估结果，而非从平均曲线计算）
    # FREEDOM 的 AUC/AUPR
    freedom_test_aucs = [r['freedom_test']['AUC'] for r in all_fold_results]
    freedom_test_auprs = [r['freedom_test']['AUPR'] for r in all_fold_results]
    freedom_mean_auc = np.mean(freedom_test_aucs)
    freedom_mean_aupr = np.mean(freedom_test_auprs)

    # MLP 的 AUC/AUPR
    mlp_test_aucs = [r['mlp_test']['AUC'] for r in all_fold_results]
    mlp_test_auprs = [r['mlp_test']['AUPR'] for r in all_fold_results]
    mlp_mean_auc = np.mean(mlp_test_aucs)
    mlp_mean_aupr = np.mean(mlp_test_auprs)

    logger.info(f'FREEDOM - Test AUC: {freedom_mean_auc:.4f}±{np.std(freedom_test_aucs):.4f}, AUPR: {freedom_mean_aupr:.4f}±{np.std(freedom_test_auprs):.4f}')
    logger.info(f'MLP - Test AUC: {mlp_mean_auc:.4f}±{np.std(mlp_test_aucs):.4f}, AUPR: {mlp_mean_aupr:.4f}±{np.std(mlp_test_auprs):.4f}')

    # 保存 FREEDOM AUC 数据（包含真实的 AUC 值）
    freedom_auc_path = os.path.join(output_dir, f'FREEDOM_{dataset_name}_5fold_AUC.h5')
    with h5py.File(freedom_auc_path, 'w') as f:
        f.create_dataset('fpr', data=mean_freedom_fpr)
        f.create_dataset('tpr', data=mean_freedom_tpr)
        f.create_dataset('auc', data=freedom_mean_auc)  # 保存真实 AUC 值
        f.create_dataset('auc_per_fold', data=np.array(freedom_test_aucs))  # 各 fold 的 AUC
    logger.info(f'Saved FREEDOM AUC data to {freedom_auc_path}')

    # 保存 FREEDOM AUPR 数据（包含真实的 AUPR 值）
    freedom_aupr_path = os.path.join(output_dir, f'FREEDOM_{dataset_name}_5fold_AUPR.h5')
    with h5py.File(freedom_aupr_path, 'w') as f:
        f.create_dataset('precision', data=mean_freedom_precision)
        f.create_dataset('recall', data=mean_freedom_recall)
        f.create_dataset('aupr', data=freedom_mean_aupr)  # 保存真实 AUPR 值
        f.create_dataset('aupr_per_fold', data=np.array(freedom_test_auprs))  # 各 fold 的 AUPR
    logger.info(f'Saved FREEDOM AUPR data to {freedom_aupr_path}')

    # 保存 MLP AUC 数据（包含真实的 AUC 值）
    mlp_auc_path = os.path.join(output_dir, f'MLP_{dataset_name}_5fold_AUC.h5')
    with h5py.File(mlp_auc_path, 'w') as f:
        f.create_dataset('fpr', data=mean_mlp_fpr)
        f.create_dataset('tpr', data=mean_mlp_tpr)
        f.create_dataset('auc', data=mlp_mean_auc)  # 保存真实 AUC 值
        f.create_dataset('auc_per_fold', data=np.array(mlp_test_aucs))  # 各 fold 的 AUC
    logger.info(f'Saved MLP AUC data to {mlp_auc_path}')

    # 保存 MLP AUPR 数据（包含真实的 AUPR 值）
    mlp_aupr_path = os.path.join(output_dir, f'MLP_{dataset_name}_5fold_AUPR.h5')
    with h5py.File(mlp_aupr_path, 'w') as f:
        f.create_dataset('precision', data=mean_mlp_precision)
        f.create_dataset('recall', data=mean_mlp_recall)
        f.create_dataset('aupr', data=mlp_mean_aupr)  # 保存真实 AUPR 值
        f.create_dataset('aupr_per_fold', data=np.array(mlp_test_auprs))  # 各 fold 的 AUPR
    logger.info(f'Saved MLP AUPR data to {mlp_aupr_path}')
    
    # 返回汇总结果
    summary = {
        'freedom': {
            metric: {
                'valid_mean': np.mean([r['freedom_valid'][metric] for r in all_fold_results]),
                'valid_std': np.std([r['freedom_valid'][metric] for r in all_fold_results]),
                'test_mean': np.mean([r['freedom_test'][metric] for r in all_fold_results]),
                'test_std': np.std([r['freedom_test'][metric] for r in all_fold_results])
            }
            for metric in metrics
        },
        'mlp': {
            metric: {
                'valid_mean': np.mean([r['mlp_valid'][metric] for r in all_fold_results]),
                'valid_std': np.std([r['mlp_valid'][metric] for r in all_fold_results]),
                'test_mean': np.mean([r['mlp_test'][metric] for r in all_fold_results]),
                'test_std': np.std([r['mlp_test'][metric] for r in all_fold_results])
            }
            for metric in metrics
        }
    }
    
    return summary


if __name__ == '__main__':
    main()
