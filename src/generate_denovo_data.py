# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
Generate Denovo Data for tsRNA-disease2

为每种疾病创建一个 fold：
- train.txt: 抹除当前疾病关联后的所有关联
- test.txt: 当前疾病的所有关联
"""

import os
import pandas as pd
from collections import defaultdict


def generate_denovo_data(dataset_name='tsRNA-disease2', num_diseases=92):
    """
    为指定数据集生成 denovo 实验数据
    
    Args:
        dataset_name: 数据集名称
        num_diseases: 疾病数量（默认 92）
    """
    # 路径设置
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', dataset_name)
    output_base = os.path.join(base_path, 'data', f'denovo_{dataset_name}')
    
    # 读取交互数据
    inter_file = os.path.join(data_path, f'{dataset_name}.inter')
    print(f"Reading interaction file: {inter_file}")
    
    df = pd.read_csv(inter_file, sep='\t')
    print(f"Total interactions: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 获取 userID 和 itemID 字段
    user_field = 'userID'
    item_field = 'itemID'
    
    # 按疾病分组关联
    disease_associations = defaultdict(list)
    for _, row in df.iterrows():
        disease_id = int(row[user_field])
        tsrna_id = int(row[item_field])
        disease_associations[disease_id].append(tsrna_id)
    
    print(f"\nDisease statistics:")
    print(f"  Total diseases with associations: {len(disease_associations)}")
    
    # 统计每种疾病的关联数
    assoc_counts = {d: len(ts) for d, ts in disease_associations.items()}
    sorted_diseases = sorted(assoc_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5 diseases by associations: {sorted_diseases[:5]}")
    
    # 创建输出目录
    os.makedirs(output_base, exist_ok=True)
    print(f"\nOutput directory: {output_base}")
    
    # 为每种疾病创建 fold
    all_associations = []
    for _, row in df.iterrows():
        all_associations.append((int(row[user_field]), int(row[item_field])))
    
    print(f"\nGenerating {num_diseases} folds...")
    
    for disease_id in range(num_diseases):
        # 创建 fold 目录 (fold 编号从 1 开始，对应疾病 ID 0)
        fold_idx = disease_id + 1
        fold_dir = os.path.join(output_base, f'{dataset_name}-fold{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # 当前疾病的关联（测试集）
        test_associations = disease_associations.get(disease_id, [])
        
        # 训练集：所有其他疾病的关联
        train_associations = [
            (d, t) for d, t in all_associations if d != disease_id
        ]
        
        # 写入 train.txt
        train_file = os.path.join(fold_dir, 'train.txt')
        with open(train_file, 'w') as f:
            for d, t in train_associations:
                f.write(f"{d} {t}\n")
        
        # 写入 test.txt
        test_file = os.path.join(fold_dir, 'test.txt')
        with open(test_file, 'w') as f:
            for t in test_associations:
                f.write(f"{disease_id} {t}\n")
        
        # 打印进度
        if (fold_idx) % 10 == 0 or disease_id == num_diseases - 1:
            print(f"  Fold {fold_idx}/{num_diseases} - Disease {disease_id}: "
                  f"train={len(train_associations)}, test={len(test_associations)}")
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_base}")
    print(f"Total folds created: {num_diseases}")
    
    # 检查无关联的疾病
    no_assoc_diseases = [d for d in range(num_diseases) if d not in disease_associations]
    if no_assoc_diseases:
        print(f"\nDiseases with no associations (empty test.txt): {no_assoc_diseases}")
    
    return output_base


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Denovo Data')
    parser.add_argument('-d', '--dataset', type=str, default='tsRNA-disease2',
                        help='Dataset name')
    parser.add_argument('-n', '--num_diseases', type=int, default=92,
                        help='Number of diseases')
    
    args = parser.parse_args()
    
    generate_denovo_data(args.dataset, args.num_diseases)
