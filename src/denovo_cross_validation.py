# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
Denovo Cross Validation for FREEDOM

De Novo 疾病预测实验：
- 对于每种疾病，抹除其所有已知关联
- 在剩余数据上训练模型
- 预测该疾病的关联
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
from collections import defaultdict

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import init_seed, get_model, get_trainer, dict2str
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.binary_evaluator import BinaryEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Denovo Cross Validation for FREEDOM + MLP')
    parser.add_argument('-d', '--dataset', type=str, default='tsRNA-disease2',
                        help='Dataset name')
    parser.add_argument('--start_fold', type=int, default=1,
                        help='Starting fold (1-indexed)')
    parser.add_argument('--end_fold', type=int, default=-1,
                        help='Ending fold (1-indexed, -1 for all)')
    parser.add_argument('--num_diseases', type=int, default=92,
                        help='Total number of diseases')
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_denovo_fold_data(fold_dir):
    """
    加载单个 fold 的数据
    
    Returns:
        train_df: 训练集 DataFrame
        test_df: 测试集 DataFrame
    """
    train_file = os.path.join(fold_dir, 'train.txt')
    test_file = os.path.join(fold_dir, 'test.txt')
    
    # 读取训练集
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                train_data.append([int(parts[0]), int(parts[1])])
    
    train_df = pd.DataFrame(train_data, columns=['userID', 'itemID'])
    train_df['rating'] = 1.0
    train_df['timestamp'] = 0
    train_df['x_label'] = 0  # 全部作为训练数据
    
    # 读取测试集
    test_data = []
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_data.append([int(parts[0]), int(parts[1])])
    
    test_df = pd.DataFrame(test_data, columns=['userID', 'itemID'])
    test_df['rating'] = 1.0
    test_df['timestamp'] = 0
    test_df['x_label'] = 0
    
    return train_df, test_df


def train_denovo_fold(config, train_df, test_df, disease_id, device, fold_idx):
    """
    训练单个疾病的 denovo 实验（FREEDOM + MLP 两阶段）
    
    Args:
        config: 配置对象
        train_df: 训练集 DataFrame
        test_df: 测试集 DataFrame
        disease_id: 当前疾病 ID
        device: 计算设备
        fold_idx: fold 索引
    
    Returns:
        freedom_results: FREEDOM 评估结果
        mlp_results: MLP 评估结果
        freedom_curve: FREEDOM 曲线数据
        mlp_curve: MLP 曲线数据
    """
    logger = getLogger()
    
    # 如果测试集为空，跳过
    if len(test_df) == 0:
        logger.info(f'  Disease {disease_id}: No test associations, skipping...')
        return None, None, None, None
    
    # 创建数据集
    full_dataset = RecDataset(config)
    train_dataset = full_dataset.copy(train_df)
    
    # 获取用户和物品数量
    n_users = train_dataset.get_user_num()
    n_items = train_dataset.get_item_num()
    
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    
    logger.info(f'  Train: {len(train_dataset)}, Test: {len(test_df)}')
    
    # 创建数据加载器
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    
    # ==================== 阶段 1: FREEDOM 训练 ====================
    logger.info(f'  === Stage 1: FREEDOM Training ===')
    
    # 初始化模型
    model = get_model(config['model'])(config, train_data).to(device)
    
    epochs = config['epochs'] if 'epochs' in config else 300
    learning_rate = config['learning_rate'] if 'learning_rate' in config else 0.001
    stopping_step = config['stopping_step'] if 'stopping_step' in config else 20
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    cur_step = 0
    
    for epoch_idx in range(epochs):
        model.train()
        model.pre_epoch_processing()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, interaction in enumerate(train_data):
            optimizer.zero_grad()
            losses = model.calculate_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
            else:
                loss = losses
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        model.post_epoch_processing()
        avg_loss = total_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            cur_step = 0
        else:
            cur_step += 1
        
        if (epoch_idx + 1) % 50 == 0:
            logger.info(f'    FREEDOM Epoch {epoch_idx+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        if cur_step >= stopping_step:
            logger.info(f'    FREEDOM Early stopping at epoch {epoch_idx+1}')
            break
    
    logger.info(f'    FREEDOM Training completed, final loss: {avg_loss:.4f}')
    
    # 保存 FREEDOM 嵌入
    embeddings_path = os.path.join(config['data_path'], config['dataset'], f'freedom_embeddings_fold{fold_idx}.pt')
    model.save_embeddings(embeddings_path)
    
    # FREEDOM 评估
    binary_evaluator = BinaryEvaluator(config)
    
    # 构建当前疾病的真实标签（标签长度 = n_items = n_tsrnas）
    # 注意：userID = disease_id, itemID = tsrna_id
    disease_labels = np.zeros(n_items)
    for _, row in test_df.iterrows():
        tsrna_id = int(row[iid_field])  # itemID 字段才是 tsrna_id
        if 0 <= tsrna_id < n_items:
            disease_labels[tsrna_id] = 1.0
    
    # 获取所有用户对所有物品的预测分数
    model.eval()
    with torch.no_grad():
        all_scores = []
        for user_id in range(n_users):
            scores = model.full_sort_predict([torch.tensor([user_id]).to(device)])
            all_scores.append(scores.cpu())
        freedom_scores_matrix = torch.cat(all_scores, dim=0)
    
    # 提取当前疾病的预测分数（取行：某疾病对所有tsRNA的预测）
    disease_scores = freedom_scores_matrix[disease_id, :].cpu().numpy()
    freedom_curve = binary_evaluator.evaluate_disease_column(disease_scores, disease_labels)
    freedom_results = {
        'AUC': freedom_curve['auc'],
        'AUPR': freedom_curve['aupr'],
        'Precision': np.mean(freedom_curve['precision']),
        'Recall': np.mean(freedom_curve['recall']),
        'F1': 2 * np.mean(freedom_curve['precision']) * np.mean(freedom_curve['recall']) / 
              (np.mean(freedom_curve['precision']) + np.mean(freedom_curve['recall'])) 
              if (np.mean(freedom_curve['precision']) + np.mean(freedom_curve['recall'])) > 0 else 0.0,
        'Accuracy': np.mean(freedom_curve['accuracy'])
    }
    
    logger.info(f'    FREEDOM Results: AUC={freedom_results["AUC"]:.4f}, AUPR={freedom_results["AUPR"]:.4f}')
    
    # ==================== 阶段 2: MLP 训练 ====================
    logger.info(f'  === Stage 2: MLP Training ===')
    
    # 加载嵌入
    from models.mlp_predictor import build_mlp_predictor
    
    emb_data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    user_embeddings = emb_data['user_embeddings'].to(device)
    item_embeddings = emb_data['item_embeddings'].to(device)
    
    # 构建 MLP 模型
    mlp_model = build_mlp_predictor(config, user_embeddings, item_embeddings)
    mlp_model = mlp_model.to(device)
    
    # MLP 训练参数
    mlp_epochs = config['mlp_epochs'] if 'mlp_epochs' in config else 100
    mlp_lr = config['mlp_lr'] if 'mlp_lr' in config else 0.001
    mlp_batch_size = config['mlp_batch_size'] if 'mlp_batch_size' in config else 256
    mlp_stopping_step = config['mlp_stopping_step'] if 'mlp_stopping_step' in config else 10
    
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=mlp_lr)
    criterion = nn.BCELoss()
    
    # 准备训练数据：正样本来自训练集
    train_users = train_df[uid_field].values
    train_items = train_df[iid_field].values
    
    # 构建用户历史交互字典（用于负采样时排除）
    user_history_items = defaultdict(set)
    for i in range(len(train_users)):
        user_history_items[train_users[i]].add(train_items[i])
    
    all_items_set = set(range(n_items))
    
    best_mlp_loss = float('inf')
    mlp_cur_step = 0
    
    for epoch_idx in range(mlp_epochs):
        mlp_model.train()
        
        # 打乱数据
        indices = np.random.permutation(len(train_users))
        
        total_loss = 0.0
        num_batches = 0
        
        # 正样本训练
        for start_idx in range(0, len(train_users), mlp_batch_size):
            end_idx = min(start_idx + mlp_batch_size, len(train_users))
            batch_indices = indices[start_idx:end_idx]
            
            batch_users = torch.tensor(train_users[batch_indices], dtype=torch.long).to(device)
            batch_items = torch.tensor(train_items[batch_indices], dtype=torch.long).to(device)
            labels = torch.ones(len(batch_indices), dtype=torch.float).to(device)
            
            mlp_optimizer.zero_grad()
            predictions = mlp_model(batch_users, batch_items)
            loss = criterion(predictions, labels)
            loss.backward()
            mlp_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 负样本训练（与 5 折交叉验证一致）
        # 排除用户历史交互的 item，负样本数量为 batch_size * 10
        n_neg_samples = min(len(train_users), mlp_batch_size * 10)
        unique_users = list(user_history_items.keys())
        
        neg_users = []
        neg_items = []
        
        for user_id in unique_users[:n_neg_samples]:
            user_history = user_history_items[user_id]
            # 排除用户历史交互的 item
            candidate_items = list(all_items_set - user_history)
            if len(candidate_items) > 0:
                neg_item = np.random.choice(candidate_items)
            else:
                # 如果所有 item 都交互过，随机选择一个
                neg_item = np.random.randint(0, n_items)
            neg_users.append(user_id)
            neg_items.append(neg_item)
        
        if len(neg_users) > 0:
            batch_users = torch.tensor(neg_users, dtype=torch.long).to(device)
            batch_items = torch.tensor(neg_items, dtype=torch.long).to(device)
            labels = torch.zeros(len(neg_users), dtype=torch.float).to(device)
            
            mlp_optimizer.zero_grad()
            predictions = mlp_model(batch_users, batch_items)
            loss = criterion(predictions, labels)
            loss.backward()
            mlp_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if avg_loss < best_mlp_loss:
            best_mlp_loss = avg_loss
            mlp_cur_step = 0
        else:
            mlp_cur_step += 1
        
        if (epoch_idx + 1) % 20 == 0:
            logger.info(f'    MLP Epoch {epoch_idx+1}/{mlp_epochs}, Loss: {avg_loss:.4f}')
        
        if mlp_cur_step >= mlp_stopping_step:
            logger.info(f'    MLP Early stopping at epoch {epoch_idx+1}')
            break
    
    logger.info(f'    MLP Training completed, final loss: {avg_loss:.4f}')
    
    # MLP 评估（只对当前疾病列评估，参考 RWR_denovo.py）
    mlp_model.eval()
    with torch.no_grad():
        all_scores = []
        for user_id in range(n_users):
            scores = mlp_model.predict_all_items(user_id, device)
            all_scores.append(scores.cpu())
        mlp_scores_matrix = torch.stack(all_scores, dim=0)
    
    # 提取当前疾病的预测分数（取行：某疾病对所有tsRNA的预测）
    disease_scores = mlp_scores_matrix[disease_id, :].cpu().numpy()
    mlp_curve = binary_evaluator.evaluate_disease_column(disease_scores, disease_labels)
    mlp_results = {
        'AUC': mlp_curve['auc'],
        'AUPR': mlp_curve['aupr'],
        'Precision': np.mean(mlp_curve['precision']),
        'Recall': np.mean(mlp_curve['recall']),
        'F1': 2 * np.mean(mlp_curve['precision']) * np.mean(mlp_curve['recall']) / 
              (np.mean(mlp_curve['precision']) + np.mean(mlp_curve['recall'])) 
              if (np.mean(mlp_curve['precision']) + np.mean(mlp_curve['recall'])) > 0 else 0.0,
        'Accuracy': np.mean(mlp_curve['accuracy'])
    }
    
    logger.info(f'    MLP Results: AUC={mlp_results["AUC"]:.4f}, AUPR={mlp_results["AUPR"]:.4f}')
    
    # 清理临时嵌入文件
    if os.path.exists(embeddings_path):
        os.remove(embeddings_path)
    
    return freedom_results, mlp_results, freedom_curve, mlp_curve


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置
    config = Config('FREEDOM', args.dataset, {})
    config['device'] = device
    config['stopping_step'] = args.stopping_step
    
    # 设置 FREEDOM 模型的固定超参数
    config['dropout'] = 0.3
    config['reg_weight'] = 1e-03
    config['use_hard_neg_mixup'] = True
    config['mixup_alpha_range'] = [0.0, 0.5]
    
    # 设置 MLP 超参数
    config['mlp_epochs'] = args.mlp_epochs
    config['mlp_lr'] = args.mlp_lr
    config['mlp_batch_size'] = args.mlp_batch_size
    config['mlp_dropout'] = args.mlp_dropout
    config['mlp_stopping_step'] = args.mlp_stopping_step
    
    # 初始化日志
    init_logger(config)
    logger = getLogger()
    
    logger.info('='*60)
    logger.info('DENOVO CROSS VALIDATION FOR FREEDOM + MLP')
    logger.info('='*60)
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Start fold: {args.start_fold}, End fold: {args.end_fold}')
    
    # 设置 fold 范围
    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold > 0 else args.num_diseases
    
    # Denovo 数据目录
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    denovo_dir = os.path.join(base_path, 'data', f'denovo_{args.dataset}')
    
    logger.info(f'Denovo directory: {denovo_dir}')
    
    # 存储结果
    all_results = []
    all_curve_data = []
    
    # 遍历所有 fold
    for fold_idx in range(start_fold, end_fold + 1):
        disease_id = fold_idx - 1  # fold 从 1 开始，疾病 ID 从 0 开始
        fold_dir = os.path.join(denovo_dir, f'{args.dataset}-fold{fold_idx}')
        
        if not os.path.exists(fold_dir):
            logger.warning(f'Fold directory not found: {fold_dir}')
            continue
        
        logger.info(f'\n--- Disease {disease_id+1}/{args.num_diseases} (fold {fold_idx}) ---')
        
        # 加载数据
        train_df, test_df = load_denovo_fold_data(fold_dir)
        logger.info(f'  True associations: {len(test_df)}')
        logger.info(f'  Training samples: {len(train_df)}')
        
        # 如果测试集为空，记录并跳过
        if len(test_df) == 0:
            logger.info(f'  Disease {disease_id}: No known associations, skipping...')
            all_results.append({
                'disease_id': disease_id,
                'fold': fold_idx,
                'AUC': 0.0,
                'AUPR': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'n_test': 0
            })
            continue
        
        # 设置随机种子
        init_seed(args.seed + fold_idx)
        
        # 训练和评估（两阶段）
        try:
            freedom_results, mlp_results, freedom_curve, mlp_curve = train_denovo_fold(
                config, train_df, test_df, disease_id, device, fold_idx
            )
            
            if freedom_results is not None:
                # FREEDOM 结果
                freedom_row = {
                    'disease_id': disease_id,
                    'fold': fold_idx,
                    'stage': 'FREEDOM',
                    'AUC': freedom_results['AUC'],
                    'AUPR': freedom_results['AUPR'],
                    'Precision': freedom_results['Precision'],
                    'Recall': freedom_results['Recall'],
                    'F1': freedom_results['F1'],
                    'Accuracy': freedom_results['Accuracy'],
                    'n_test': len(test_df)
                }
                all_results.append(freedom_row)
                
                # MLP 结果
                mlp_row = {
                    'disease_id': disease_id,
                    'fold': fold_idx,
                    'stage': 'MLP',
                    'AUC': mlp_results['AUC'],
                    'AUPR': mlp_results['AUPR'],
                    'Precision': mlp_results['Precision'],
                    'Recall': mlp_results['Recall'],
                    'F1': mlp_results['F1'],
                    'Accuracy': mlp_results['Accuracy'],
                    'n_test': len(test_df)
                }
                all_results.append(mlp_row)
                
                all_curve_data.append(('FREEDOM', freedom_curve))
                all_curve_data.append(('MLP', mlp_curve))
                
                logger.info(f'  Final: FREEDOM AUC={freedom_results["AUC"]:.4f}, MLP AUC={mlp_results["AUC"]:.4f}')
            else:
                all_results.append({
                    'disease_id': disease_id,
                    'fold': fold_idx,
                    'stage': 'FREEDOM',
                    'AUC': 0.0,
                    'AUPR': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1': 0.0,
                    'Accuracy': 0.0,
                    'n_test': len(test_df)
                })
                all_results.append({
                    'disease_id': disease_id,
                    'fold': fold_idx,
                    'stage': 'MLP',
                    'AUC': 0.0,
                    'AUPR': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1': 0.0,
                    'Accuracy': 0.0,
                    'n_test': len(test_df)
                })
        except Exception as e:
            import traceback
            logger.error(f'  Error processing disease {disease_id}: {str(e)}')
            logger.error(traceback.format_exc())
            all_results.append({
                'disease_id': disease_id,
                'fold': fold_idx,
                'stage': 'FREEDOM',
                'AUC': 0.0,
                'AUPR': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'Accuracy': 0.0,
                'n_test': len(test_df)
            })
            all_results.append({
                'disease_id': disease_id,
                'fold': fold_idx,
                'stage': 'MLP',
                'AUC': 0.0,
                'AUPR': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'Accuracy': 0.0,
                'n_test': len(test_df)
            })
    
    # 汇总结果
    logger.info('\n' + '='*60)
    logger.info('DENOVO RESULTS SUMMARY')
    logger.info('='*60)
    
    # 分别处理 FREEDOM 和 MLP 结果
    freedom_results = [r for r in all_results if r.get('stage') == 'FREEDOM' and r['n_test'] > 0]
    mlp_results_list = [r for r in all_results if r.get('stage') == 'MLP' and r['n_test'] > 0]
    
    metrics = ['AUC', 'AUPR', 'Precision', 'Recall', 'F1', 'Accuracy']
    
    if len(freedom_results) > 0:
        logger.info('\n--- FREEDOM Results ---')
        for metric in metrics:
            values = [r[metric] for r in freedom_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f'{metric}: {mean_val:.4f} ± {std_val:.4f}')
    
    if len(mlp_results_list) > 0:
        logger.info('\n--- MLP Results ---')
        for metric in metrics:
            values = [r[metric] for r in mlp_results_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f'{metric}: {mean_val:.4f} ± {std_val:.4f}')
    
    # 保存结果到 CSV
    results_df = pd.DataFrame(all_results)
    output_csv = os.path.join(base_path, 'data', args.dataset, 'denovo_results.csv')
    results_df.to_csv(output_csv, index=False)
    logger.info(f'\nResults saved to: {output_csv}')
    
    # 分别保存 FREEDOM 和 MLP 曲线数据
    output_dir = os.path.join(base_path, 'tsRNA_PlotPng')
    os.makedirs(output_dir, exist_ok=True)
    
    freedom_curves = [c[1] for c in all_curve_data if c[0] == 'FREEDOM']
    mlp_curves = [c[1] for c in all_curve_data if c[0] == 'MLP']
    
    # FREEDOM 曲线
    if len(freedom_curves) > 0:
        tpr_arr = np.array([c['tpr'] for c in freedom_curves])
        fpr_arr = np.array([c['fpr'] for c in freedom_curves])
        precision_arr = np.array([c['precision'] for c in freedom_curves])
        recall_arr = np.array([c['recall'] for c in freedom_curves])
        
        mean_tpr = np.mean(tpr_arr, axis=0)
        mean_fpr = np.mean(fpr_arr, axis=0)
        mean_precision = np.mean(precision_arr, axis=0)
        mean_recall = np.mean(recall_arr, axis=0)
        
        roc_auc = np.trapz(mean_tpr, mean_fpr)
        pr_auc = np.trapz(mean_precision, mean_recall)
        
        logger.info(f'\nFREEDOM Mean ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}')
        
        auc_path = os.path.join(output_dir, f'FREEDOM_{args.dataset}_denovo_AUC.h5')
        with h5py.File(auc_path, 'w') as f:
            f.create_dataset('fpr', data=mean_fpr)
            f.create_dataset('tpr', data=mean_tpr)
        logger.info(f'FREEDOM AUC curve saved to: {auc_path}')
        
        aupr_path = os.path.join(output_dir, f'FREEDOM_{args.dataset}_denovo_AUPR.h5')
        with h5py.File(aupr_path, 'w') as f:
            f.create_dataset('precision', data=mean_precision)
            f.create_dataset('recall', data=mean_recall)
        logger.info(f'FREEDOM AUPR curve saved to: {aupr_path}')
    
    # MLP 曲线
    if len(mlp_curves) > 0:
        tpr_arr = np.array([c['tpr'] for c in mlp_curves])
        fpr_arr = np.array([c['fpr'] for c in mlp_curves])
        precision_arr = np.array([c['precision'] for c in mlp_curves])
        recall_arr = np.array([c['recall'] for c in mlp_curves])
        
        mean_tpr = np.mean(tpr_arr, axis=0)
        mean_fpr = np.mean(fpr_arr, axis=0)
        mean_precision = np.mean(precision_arr, axis=0)
        mean_recall = np.mean(recall_arr, axis=0)
        
        roc_auc = np.trapz(mean_tpr, mean_fpr)
        pr_auc = np.trapz(mean_precision, mean_recall)
        
        logger.info(f'\nMLP Mean ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}')
        
        auc_path = os.path.join(output_dir, f'MLP_{args.dataset}_denovo_AUC.h5')
        with h5py.File(auc_path, 'w') as f:
            f.create_dataset('fpr', data=mean_fpr)
            f.create_dataset('tpr', data=mean_tpr)
        logger.info(f'MLP AUC curve saved to: {auc_path}')
        
        aupr_path = os.path.join(output_dir, f'MLP_{args.dataset}_denovo_AUPR.h5')
        with h5py.File(aupr_path, 'w') as f:
            f.create_dataset('precision', data=mean_precision)
            f.create_dataset('recall', data=mean_recall)
        logger.info(f'MLP AUPR curve saved to: {aupr_path}')
    
    logger.info('\nDenovo experiment completed!')


if __name__ == '__main__':
    main()
