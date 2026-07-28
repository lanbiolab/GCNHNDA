"""
处理 tsRNA_disease_data_v3.xlsx 数据，转换为 FREEDOM 模型所需格式

映射关系：
- disease → userID (新建立映射)
- tsRNA → itemID (使用 tsrna_names_dedup.json)

输出文件：
- tsRNA-disease2.inter: 交互数据
- u_id_mapping.csv: disease → userID 映射
- i_id_mapping.csv: tsRNA → itemID 映射
- text_feat.npy: tsRNA 文本嵌入
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
import random

# 设置随机种子以保证可重复性
random.seed(42)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN_DIR = os.path.join(BASE_DIR, 'tsRNA-origin')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'tsRNA-disease2')

# 输入文件
EXCEL_FILE = os.path.join(ORIGIN_DIR, 'tsRNA_disease_data_v3.xlsx')
TSRNA_MAPPING_FILE = os.path.join(ORIGIN_DIR, 'tsrna_names_dedup.json')
EMBEDDING_FILE = os.path.join(ORIGIN_DIR, 'embeddings_dedup.npy')

# 输出文件
INTER_FILE = os.path.join(OUTPUT_DIR, 'tsRNA-disease2.inter')
U_ID_MAPPING_FILE = os.path.join(OUTPUT_DIR, 'u_id_mapping.csv')
I_ID_MAPPING_FILE = os.path.join(OUTPUT_DIR, 'i_id_mapping.csv')
TEXT_FEAT_FILE = os.path.join(OUTPUT_DIR, 'text_feat.npy')

# 数据划分比例 (训练:验证:测试 = 16:4:5)
TRAIN_RATIO = 16
VALID_RATIO = 4
TEST_RATIO = 5


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("处理 tsRNA-disease 数据 v3")
    print("=" * 60)
    
    # 1. 读取 Excel 数据
    print("\n[1] 读取 Excel 数据...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"    总行数: {len(df)}")
    print(f"    列名: {list(df.columns)}")
    
    # 2. 读取 tsRNA 映射
    print("\n[2] 读取 tsRNA ID 映射...")
    with open(TSRNA_MAPPING_FILE, 'r') as f:
        tsrna_to_itemid = json.load(f)
    print(f"    tsRNA 数量: {len(tsrna_to_itemid)}")
    
    # 3. 建立 disease → userID 映射
    print("\n[3] 建立 disease → userID 映射...")
    unique_diseases = df['Disease_Name'].unique()
    disease_to_userid = {d: i for i, d in enumerate(unique_diseases)}
    print(f"    Disease 数量: {len(unique_diseases)}")
    
    # 4. 生成交互数据
    print("\n[4] 生成交互数据...")
    interactions = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        disease_name = row['Disease_Name']
        tsrna_name = row['tsRNA_Name']
        
        # 检查 tsRNA 是否在映射中
        if tsrna_name not in tsrna_to_itemid:
            skipped_count += 1
            continue
        
        user_id = disease_to_userid[disease_name]
        item_id = tsrna_to_itemid[tsrna_name]
        
        interactions.append({
            'userID': user_id,
            'itemID': item_id,
            'rating': 1.0,
            'timestamp': 0,
            'disease_name': disease_name  # 用于分组划分
        })
    
    print(f"    有效交互数: {len(interactions)}")
    print(f"    跳过的交互数 (tsRNA不在映射中): {skipped_count}")
    
    # 5. 按 disease 分组，按比例划分 x_label
    print("\n[5] 数据划分 (训练:验证:测试 = 16:4:5)...")
    
    # 按 disease 分组
    disease_interactions = defaultdict(list)
    for inter in interactions:
        disease_interactions[inter['disease_name']].append(inter)
    
    total_train, total_valid, total_test = 0, 0, 0
    
    for disease_name, inters in disease_interactions.items():
        n = len(inters)
        random.shuffle(inters)
        
        # 计算划分点
        train_end = int(n * TRAIN_RATIO / (TRAIN_RATIO + VALID_RATIO + TEST_RATIO))
        valid_end = int(n * (TRAIN_RATIO + VALID_RATIO) / (TRAIN_RATIO + VALID_RATIO + TEST_RATIO))
        
        for i, inter in enumerate(inters):
            if i < train_end:
                inter['x_label'] = 0  # 训练集
                total_train += 1
            elif i < valid_end:
                inter['x_label'] = 1  # 验证集
                total_valid += 1
            else:
                inter['x_label'] = 2  # 测试集
                total_test += 1
    
    print(f"    训练集: {total_train}")
    print(f"    验证集: {total_valid}")
    print(f"    测试集: {total_test}")
    
    # 6. 保存交互文件
    print("\n[6] 保存交互文件...")
    inter_df = pd.DataFrame(interactions)
    inter_df = inter_df[['userID', 'itemID', 'rating', 'timestamp', 'x_label']]
    inter_df.to_csv(INTER_FILE, sep='\t', index=False)
    print(f"    保存到: {INTER_FILE}")
    
    # 7. 保存 disease → userID 映射
    print("\n[7] 保存 disease → userID 映射...")
    u_mapping_df = pd.DataFrame([
        {'disease_id': d, 'userID': uid}
        for d, uid in disease_to_userid.items()
    ])
    u_mapping_df.to_csv(U_ID_MAPPING_FILE, index=False)
    print(f"    保存到: {U_ID_MAPPING_FILE}")
    
    # 8. 保存 tsRNA → itemID 映射
    print("\n[8] 保存 tsRNA → itemID 映射...")
    i_mapping_df = pd.DataFrame([
        {'tsRNA_id': t, 'itemID': iid}
        for t, iid in tsrna_to_itemid.items()
    ])
    # 按 itemID 排序
    i_mapping_df = i_mapping_df.sort_values('itemID').reset_index(drop=True)
    i_mapping_df.to_csv(I_ID_MAPPING_FILE, index=False)
    print(f"    保存到: {I_ID_MAPPING_FILE}")
    
    # 9. 复制文本嵌入
    print("\n[9] 复制文本嵌入...")
    embeddings = np.load(EMBEDDING_FILE)
    np.save(TEXT_FEAT_FILE, embeddings)
    print(f"    嵌入形状: {embeddings.shape}")
    print(f"    保存到: {TEXT_FEAT_FILE}")
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"  - 用户数 (disease): {len(disease_to_userid)}")
    print(f"  - 物品数 (tsRNA): {len(tsrna_to_itemid)}")
    print(f"  - 交互数: {len(interactions)}")
    print(f"  - 嵌入维度: {embeddings.shape[1]}")


if __name__ == '__main__':
    main()
