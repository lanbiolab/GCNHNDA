# coding: utf-8
"""
tsRNA-Disease 数据处理脚本
将 Excel 数据转换为 FREEDOM 推荐系统格式
"""

import os
import pandas as pd
import numpy as np
import h5py
from collections import defaultdict


def load_h5_ids(h5_path, id_key):
    """从 h5 文件中加载 ID 列表"""
    with h5py.File(h5_path, 'r') as f:
        ids = f[id_key][:]
        # 将 bytes 转换为字符串
        ids = [x.decode('utf-8') if isinstance(x, bytes) else x for x in ids]
    return ids


def load_embeddings(h5_path, embed_key='embeddings'):
    """从 h5 文件中加载 embeddings"""
    with h5py.File(h5_path, 'r') as f:
        embeddings = f[embed_key][:]
    return embeddings


def create_id_mapping(ids):
    """创建 ID 到索引的映射字典"""
    return {id_: idx for idx, id_ in enumerate(ids)}


def assign_x_label(n_samples, ratios=(16, 4, 5)):
    """
    按比例分配 x_label
    ratios: (训练集, 验证集, 测试集) 比例
    返回: x_label 列表 (0=训练, 1=验证, 2=测试)
    """
    total = sum(ratios)
    n_train = int(n_samples * ratios[0] / total)
    n_val = int(n_samples * ratios[1] / total)
    # 测试集取剩余部分，避免舍入误差
    n_test = n_samples - n_train - n_val
    
    x_labels = [0] * n_train + [1] * n_val + [2] * n_test
    
    # 随机打乱
    np.random.seed(42)
    np.random.shuffle(x_labels)
    
    return x_labels


def process_tsRNA_disease_data():
    """主处理函数"""
    
    # 路径设置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tsRNA_origin_dir = os.path.join(base_dir, 'tsRNA-origin')
    data_dir = os.path.join(base_dir, 'data', 'tsRNA-disease')
    
    # 创建输出目录
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("tsRNA-Disease 数据处理")
    print("=" * 60)
    
    # 1. 加载 h5 文件中的 ID 映射
    print("\n[1/6] 加载 h5 文件中的 ID 映射...")
    
    tsRNA_h5_path = os.path.join(tsRNA_origin_dir, 'tsrna_embeddings.h5')
    disease_h5_path = os.path.join(tsRNA_origin_dir, 'disease_embeddings.h5')
    
    tsRNA_ids = load_h5_ids(tsRNA_h5_path, 'tsrna_ids')
    disease_ids = load_h5_ids(disease_h5_path, 'disease_ids')
    
    print(f"  - tsRNA IDs: {len(tsRNA_ids)} 个")
    print(f"  - Disease IDs: {len(disease_ids)} 个")
    
    # 创建 ID 映射字典
    tsRNA_to_userID = create_id_mapping(tsRNA_ids)
    disease_to_itemID = create_id_mapping(disease_ids)
    
    # 2. 加载疾病 embeddings
    print("\n[2/6] 加载 Disease embeddings...")
    disease_embeddings = load_embeddings(disease_h5_path, 'embeddings')
    print(f"  - Disease embeddings shape: {disease_embeddings.shape}")
    
    # 3. 读取 Excel 文件
    print("\n[3/6] 读取 Excel 交互数据...")
    excel_path = os.path.join(tsRNA_origin_dir, 'tsRNA-disease数据-20251012_with_descriptions_new.xlsx')
    df = pd.read_excel(excel_path)
    print(f"  - Excel 数据行数: {len(df)}")
    print(f"  - 列名: {list(df.columns)}")
    
    # 4. 生成交互数据
    print("\n[4/6] 生成交互数据文件 (tsRNA-disease.inter)...")
    
    interactions = []
    invalid_count = 0
    
    for idx, row in df.iterrows():
        tsRNA_id = row['tsRNA ID']
        disease_id = row['disease ID']
        
        # 检查 ID 是否在映射中
        if tsRNA_id not in tsRNA_to_userID:
            print(f"  警告: tsRNA ID '{tsRNA_id}' 不在 h5 文件中，跳过")
            invalid_count += 1
            continue
        if disease_id not in disease_to_itemID:
            print(f"  警告: Disease ID '{disease_id}' 不在 h5 文件中，跳过")
            invalid_count += 1
            continue
        
        userID = tsRNA_to_userID[tsRNA_id]
        itemID = disease_to_itemID[disease_id]
        
        interactions.append({
            'userID': userID,
            'itemID': itemID,
            'tsRNA_id': tsRNA_id,
            'disease_id': disease_id
        })
    
    print(f"  - 有效交互数: {len(interactions)}")
    if invalid_count > 0:
        print(f"  - 跳过无效数据: {invalid_count} 条")
    
    # 分配 x_label
    x_labels = assign_x_label(len(interactions), ratios=(16, 4, 5))
    
    # 生成 inter 文件内容
    inter_lines = ['userID\titemID\trating\ttimestamp\tx_label']
    timestamp = 0  # 固定时间戳
    
    for i, inter in enumerate(interactions):
        line = f"{inter['userID']}\t{inter['itemID']}\t1.0\t{timestamp}\t{x_labels[i]}"
        inter_lines.append(line)
    
    # 保存 inter 文件
    inter_path = os.path.join(data_dir, 'tsRNA-disease.inter')
    with open(inter_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(inter_lines))
    print(f"  - 已保存: {inter_path}")
    
    # 统计 x_label 分布
    x_label_counts = {0: 0, 1: 0, 2: 0}
    for x in x_labels:
        x_label_counts[x] += 1
    print(f"  - x_label 分布: 训练集(0)={x_label_counts[0]}, 验证集(1)={x_label_counts[1]}, 测试集(2)={x_label_counts[2]}")
    
    # 5. 生成映射文件
    print("\n[5/6] 生成 ID 映射文件...")
    
    # u_id_mapping.csv (tsRNA -> userID)
    u_mapping_lines = ['tsRNA_id\tuserID']
    for tsRNA_id in tsRNA_ids:
        userID = tsRNA_to_userID[tsRNA_id]
        u_mapping_lines.append(f"{tsRNA_id}\t{userID}")
    
    u_mapping_path = os.path.join(data_dir, 'u_id_mapping.csv')
    with open(u_mapping_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(u_mapping_lines))
    print(f"  - 已保存: {u_mapping_path} ({len(tsRNA_ids)} 条)")
    
    # i_id_mapping.csv (disease -> itemID)
    i_mapping_lines = ['disease_id\titemID']
    for disease_id in disease_ids:
        itemID = disease_to_itemID[disease_id]
        i_mapping_lines.append(f"{disease_id}\t{itemID}")
    
    i_mapping_path = os.path.join(data_dir, 'i_id_mapping.csv')
    with open(i_mapping_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(i_mapping_lines))
    print(f"  - 已保存: {i_mapping_path} ({len(disease_ids)} 条)")
    
    # 6. 生成 user_graph_dict.npy
    print("\n[6/6] 生成用户图和特征文件...")
    
    # 构建用户图: {userID: [[itemID列表], [权重列表]]}
    user_graph = defaultdict(lambda: [[], []])
    
    for inter in interactions:
        userID = inter['userID']
        itemID = inter['itemID']
        user_graph[userID][0].append(itemID)
        user_graph[userID][1].append(1.0)  # 权重为 1.0
    
    # 转换为普通 dict 并保存
    user_graph_dict = dict(user_graph)
    user_graph_path = os.path.join(data_dir, 'user_graph_dict.npy')
    np.save(user_graph_path, user_graph_dict)
    print(f"  - 已保存: {user_graph_path} ({len(user_graph_dict)} 个用户)")
    
    # 保存 text_feat.npy (disease embeddings)
    text_feat_path = os.path.join(data_dir, 'text_feat.npy')
    np.save(text_feat_path, disease_embeddings)
    print(f"  - 已保存: {text_feat_path} {disease_embeddings.shape}")
    
    # 7. 完成
    print("\n" + "=" * 60)
    print("数据处理完成!")
    print("=" * 60)
    print(f"\n输出目录: {data_dir}")
    print("\n生成的文件:")
    print(f"  1. tsRNA-disease.inter    - 交互数据 ({len(interactions)} 条)")
    print(f"  2. u_id_mapping.csv       - tsRNA ID 映射 ({len(tsRNA_ids)} 条)")
    print(f"  3. i_id_mapping.csv       - Disease ID 映射 ({len(disease_ids)} 条)")
    print(f"  4. user_graph_dict.npy    - 用户交互图 ({len(user_graph_dict)} 个用户)")
    print(f"  5. text_feat.npy          - Disease 文本特征 {disease_embeddings.shape}")
    
    # 数据统计
    print("\n数据统计:")
    print(f"  - 用户数 (tsRNA): {len(tsRNA_ids)}")
    print(f"  - 物品数 (Disease): {len(disease_ids)}")
    print(f"  - 交互数: {len(interactions)}")
    print(f"  - 训练集: {x_label_counts[0]} 条 ({x_label_counts[0]/len(interactions)*100:.1f}%)")
    print(f"  - 验证集: {x_label_counts[1]} 条 ({x_label_counts[1]/len(interactions)*100:.1f}%)")
    print(f"  - 测试集: {x_label_counts[2]} 条 ({x_label_counts[2]/len(interactions)*100:.1f}%)")


if __name__ == '__main__':
    process_tsRNA_disease_data()
