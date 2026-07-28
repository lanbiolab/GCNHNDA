# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
Case Study: tsRNA-Disease Association Prediction

案例分析实验：
- 用所有数据训练 FREEDOM + MLP 模型
- 预测指定疾病（如 Breast Neoplasms）与所有 tsRNA 的关联分数
- 按 MLP 分数排名，统计 Top-K 中已知关联的数量
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from time import time
from logging import getLogger
from collections import defaultdict

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import init_seed, get_model
from utils.dataloader import TrainDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Case Study for tsRNA-Disease Association')
    parser.add_argument('-d', '--dataset', type=str, default='tsRNA-disease2',
                        help='Dataset name')
    parser.add_argument('--disease', type=str, default='Breast Neoplasms',
                        help='Target disease name for case study')
    parser.add_argument('--top_k', type=int, nargs='+', default=[10, 20],
                        help='Top-K values for hit calculation')
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


def load_all_data(config):
    """
    加载所有数据（不做 train/test 分割）
    
    Returns:
        all_df: 所有交互数据 DataFrame
        n_users: 用户数量
        n_items: 物品数量
    """
    dataset = RecDataset(config)
    all_df = dataset.df.copy()
    n_users = dataset.get_user_num()
    n_items = dataset.get_item_num()
    return all_df, n_users, n_items


def load_id_mappings(data_path, dataset_name):
    """
    加载 ID 映射文件
    
    Returns:
        user_id_to_name: userID -> disease_name
        user_name_to_id: disease_name -> userID
        item_id_to_name: itemID -> tsRNA_name
        item_name_to_id: tsRNA_name -> itemID
    """
    # 疾病映射
    user_mapping_path = os.path.join(data_path, dataset_name, 'u_id_mapping.csv')
    user_mapping_df = pd.read_csv(user_mapping_path)
    user_id_to_name = dict(zip(user_mapping_df['userID'], user_mapping_df['disease_id']))
    user_name_to_id = dict(zip(user_mapping_df['disease_id'], user_mapping_df['userID']))
    
    # tsRNA 映射
    item_mapping_path = os.path.join(data_path, dataset_name, 'i_id_mapping.csv')
    item_mapping_df = pd.read_csv(item_mapping_path)
    item_id_to_name = dict(zip(item_mapping_df['itemID'], item_mapping_df['tsRNA_id']))
    item_name_to_id = dict(zip(item_mapping_df['tsRNA_id'], item_mapping_df['itemID']))
    
    return user_id_to_name, user_name_to_id, item_id_to_name, item_name_to_id


def get_known_associations(all_df, disease_id, iid_field):
    """
    获取指定疾病的已知关联 tsRNA 集合
    
    Returns:
        known_items: 已知关联的 itemID 集合
    """
    uid_field = 'userID'  # 数据集中的用户ID字段名
    disease_df = all_df[all_df[uid_field] == disease_id]
    known_items = set(disease_df[iid_field].values)
    return known_items


def train_freedom(config, train_data, device, logger):
    """
    训练 FREEDOM 模型
    
    Returns:
        model: 训练好的 FREEDOM 模型
    """
    logger.info('=== Stage 1: FREEDOM Training ===')
    
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
            logger.info(f'  FREEDOM Epoch {epoch_idx+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        if cur_step >= stopping_step:
            logger.info(f'  FREEDOM Early stopping at epoch {epoch_idx+1}')
            break
    
    logger.info(f'  FREEDOM Training completed, final loss: {avg_loss:.4f}')
    return model


def train_mlp(config, model, train_data, n_users, n_items, device, logger):
    """
    训练 MLP 分类器
    
    Returns:
        mlp_model: 训练好的 MLP 模型
        user_embeddings: 用户嵌入
        item_embeddings: 物品嵌入
    """
    logger.info('=== Stage 2: MLP Training ===')
    
    # 获取 FREEDOM 嵌入
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.forward(model.norm_adj)
    user_embeddings = user_embeddings.detach()
    item_embeddings = item_embeddings.detach()
    
    # 构建 MLP 模型
    from models.mlp_predictor import build_mlp_predictor
    mlp_model = build_mlp_predictor(config, user_embeddings, item_embeddings)
    mlp_model = mlp_model.to(device)
    
    # MLP 训练参数
    mlp_epochs = config['mlp_epochs'] if 'mlp_epochs' in config else 100
    mlp_lr = config['mlp_lr'] if 'mlp_lr' in config else 0.001
    mlp_batch_size = config['mlp_batch_size'] if 'mlp_batch_size' in config else 256
    mlp_stopping_step = config['mlp_stopping_step'] if 'mlp_stopping_step' in config else 10
    
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=mlp_lr)
    criterion = nn.BCELoss()
    
    # 准备训练数据
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    all_df = train_data.dataset.df
    train_users = all_df[uid_field].values
    train_items = all_df[iid_field].values
    
    # 构建用户历史交互字典
    user_history_items = defaultdict(set)
    for i in range(len(train_users)):
        user_history_items[train_users[i]].add(train_items[i])
    
    all_items_set = set(range(n_items))
    
    best_mlp_loss = float('inf')
    mlp_cur_step = 0
    
    for epoch_idx in range(mlp_epochs):
        mlp_model.train()
        
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
        
        # 负样本训练
        n_neg_samples = min(len(train_users), mlp_batch_size * 10)
        unique_users = list(user_history_items.keys())
        
        neg_users = []
        neg_items = []
        
        for user_id in unique_users[:n_neg_samples]:
            user_history = user_history_items[user_id]
            candidate_items = list(all_items_set - user_history)
            if len(candidate_items) > 0:
                neg_item = np.random.choice(candidate_items)
            else:
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
            logger.info(f'  MLP Epoch {epoch_idx+1}/{mlp_epochs}, Loss: {avg_loss:.4f}')
        
        if mlp_cur_step >= mlp_stopping_step:
            logger.info(f'  MLP Early stopping at epoch {epoch_idx+1}')
            break
    
    logger.info(f'  MLP Training completed, final loss: {avg_loss:.4f}')
    return mlp_model, user_embeddings, item_embeddings


def predict_and_rank(model, mlp_model, disease_id, n_users, n_items, device):
    """
    预测指定疾病对所有 tsRNA 的关联分数并排名
    
    Returns:
        freedom_scores: FREEDOM 预测分数数组
        mlp_scores: MLP 预测分数数组
        mlp_ranking: MLP 分数排名（降序）
    """
    model.eval()
    mlp_model.eval()
    
    with torch.no_grad():
        # FREEDOM 预测
        freedom_scores = model.full_sort_predict([torch.tensor([disease_id]).to(device)])
        freedom_scores = freedom_scores.cpu().numpy().flatten()
        
        # MLP 预测
        mlp_scores = mlp_model.predict_all_items(disease_id, device).cpu().numpy()
    
    # 按 MLP 分数降序排名
    mlp_ranking = np.argsort(-mlp_scores)
    
    return freedom_scores, mlp_scores, mlp_ranking


def print_top_k_results(mlp_ranking, mlp_scores, known_items, item_id_to_name, top_k_list, logger):
    """
    打印 Top-K 结果并统计命中率
    """
    for k in top_k_list:
        logger.info(f'\n--- Top-{k} Predictions ---')
        logger.info(f"{'Rank':<6} {'tsRNA_id':<30} {'MLP_Score':<12} {'Known'}")
        logger.info('-' * 60)
        
        hits = 0
        for rank, item_idx in enumerate(mlp_ranking[:k]):
            tsrna_name = item_id_to_name.get(item_idx, f'item_{item_idx}')
            score = mlp_scores[item_idx]
            is_known = item_idx in known_items
            if is_known:
                hits += 1
            known_str = 'Yes ✓' if is_known else 'No'
            logger.info(f'{rank+1:<6} {tsrna_name:<30} {score:<12.4f} {known_str}')
        
        hit_rate = hits / k * 100
        logger.info(f'\nTop-{k} hits: {hits}/{k} ({hit_rate:.1f}%)')


def save_results(mlp_ranking, freedom_scores, mlp_scores, known_items, item_id_to_name, 
                 disease_name, output_path, logger):
    """
    保存完整排名结果到 CSV
    """
    results = []
    for rank, item_idx in enumerate(mlp_ranking):
        tsrna_name = item_id_to_name.get(item_idx, f'item_{item_idx}')
        is_known = item_idx in known_items
        results.append({
            'rank': rank + 1,
            'tsRNA_id': tsrna_name,
            'tsRNA_index': item_idx,
            'freedom_score': freedom_scores[item_idx],
            'mlp_score': mlp_scores[item_idx],
            'is_known': is_known
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f'\nFull ranking saved to: {output_path}')
    
    return results_df


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
    config['stopping_step'] = args.stopping_step
    config['epochs'] = args.freedom_epochs
    config['learning_rate'] = args.freedom_lr
    
    # FREEDOM 固定超参数
    config['dropout'] = 0.3
    config['reg_weight'] = 1e-03
    config['use_hard_neg_mixup'] = True
    config['mixup_alpha_range'] = [0.0, 0.5]
    
    # MLP 超参数
    config['mlp_epochs'] = args.mlp_epochs
    config['mlp_lr'] = args.mlp_lr
    config['mlp_batch_size'] = args.mlp_batch_size
    config['mlp_dropout'] = args.mlp_dropout
    config['mlp_stopping_step'] = args.mlp_stopping_step
    
    # 初始化日志
    init_logger(config)
    logger = getLogger()
    
    # 获取基础路径
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取数据路径（从 config 中获取，Config 会自动设置）
    data_path = config['data_path']
    
    logger.info('='*60)
    logger.info('CASE STUDY: tsRNA-Disease Association Prediction')
    logger.info('='*60)
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Target disease: {args.disease}')
    logger.info(f'Top-K values: {args.top_k}')
    
    # 加载 ID 映射
    user_id_to_name, user_name_to_id, item_id_to_name, item_name_to_id = load_id_mappings(
        data_path, args.dataset
    )
    
    # 获取目标疾病 ID
    if args.disease not in user_name_to_id:
        logger.error(f"Disease '{args.disease}' not found in dataset!")
        logger.info(f"Available diseases: {list(user_name_to_id.keys())[:10]}...")
        return
    
    disease_id = user_name_to_id[args.disease]
    logger.info(f"Disease ID: {disease_id}")
    
    # 加载所有数据
    logger.info('\nLoading all data...')
    all_df, n_users, n_items = load_all_data(config)
    logger.info(f'Total interactions: {len(all_df)}')
    logger.info(f'Users (diseases): {n_users}, Items (tsRNAs): {n_items}')
    
    # 获取已知关联
    known_items = get_known_associations(all_df, disease_id, config['ITEM_ID_FIELD'])
    logger.info(f'Known associations for {args.disease}: {len(known_items)} tsRNAs')
    
    # 创建训练数据加载器
    # 将所有数据标记为训练数据
    all_df['x_label'] = 0
    full_dataset = RecDataset(config)
    train_dataset = full_dataset.copy(all_df)
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    
    # 训练 FREEDOM
    model = train_freedom(config, train_data, device, logger)
    
    # 训练 MLP
    mlp_model, user_emb, item_emb = train_mlp(config, model, train_data, n_users, n_items, device, logger)
    
    # 预测和排名
    logger.info('\n=== Stage 3: Prediction and Ranking ===')
    freedom_scores, mlp_scores, mlp_ranking = predict_and_rank(
        model, mlp_model, disease_id, n_users, n_items, device
    )
    
    # 打印 Top-K 结果
    print_top_k_results(mlp_ranking, mlp_scores, known_items, item_id_to_name, args.top_k, logger)
    
    # 保存完整结果
    output_dir = os.path.join(data_path, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    disease_name_safe = args.disease.replace(' ', '_').replace(',', '')
    output_path = os.path.join(output_dir, f'case_study_{disease_name_safe}_ranking.csv')
    
    save_results(mlp_ranking, freedom_scores, mlp_scores, known_items, item_id_to_name,
                 args.disease, output_path, logger)
    
    logger.info('\n' + '='*60)
    logger.info('Case study completed!')
    logger.info('='*60)


if __name__ == '__main__':
    main()
