# coding: utf-8
"""
Binary Classification Evaluator for tsRNA-Disease Recommendation
计算 AUC, AUPR, Precision, Recall, F1, Accuracy 指标
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


class BinaryEvaluator:
    """二分类评估器"""
    
    def __init__(self, config):
        self.config = config
        
    def evaluate(self, scores_matrix, labels_matrix):
        """
        对所有 260x53 的 tsRNA-disease 组合进行二分类评估
        
        Args:
            scores_matrix: [n_users, n_items] 预测分数矩阵
            labels_matrix: [n_users, n_items] 标签矩阵 (0/1)
        
        Returns:
            dict: 包含 AUC, AUPR, Precision, Recall, F1 的字典
        """
        # 展平为 1D 数组
        scores = scores_matrix.flatten().cpu().numpy()  # [n_users * n_items]
        labels = labels_matrix.flatten().cpu().numpy()   # [n_users * n_items]
        
        # 计算 AUC 和 AUPR
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0  # 如果只有一个类别，AUC 无法计算
            
        try:
            aupr = average_precision_score(labels, scores)
        except ValueError:
            aupr = 0.0
        
        # 遍历所有阈值计算 Precision, Recall, F1, Accuracy
        precisions, recalls, f1s, accuracies = self._calculate_metrics_at_thresholds(scores, labels)
        
        # 计算平均指标
        avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
        avg_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
        avg_f1 = np.mean(f1s) if len(f1s) > 0 else 0.0
        avg_accuracy = np.mean(accuracies) if len(accuracies) > 0 else 0.0
        
        return {
            'AUC': auc,
            'AUPR': aupr,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1': avg_f1,
            'Accuracy': avg_accuracy
        }
    
    def _calculate_metrics_at_thresholds(self, scores, labels):
        """
        对所有可能的阈值计算 Precision, Recall, F1, Accuracy
        
        Args:
            scores: [N] 预测分数数组
            labels: [N] 标签数组 (0/1)
        
        Returns:
            precisions, recalls, f1s, accuracies 列表
        """
        # 获取唯一的分数作为阈值（排序后）
        unique_scores = np.unique(scores)
        
        # 限制阈值数量以提高效率（如果分数太多）
        max_thresholds = 1000
        if len(unique_scores) > max_thresholds:
            indices = np.linspace(0, len(unique_scores) - 1, max_thresholds, dtype=int)
            thresholds = unique_scores[indices]
        else:
            thresholds = unique_scores
        
        precisions = []
        recalls = []
        f1s = []
        accuracies = []
        
        for threshold in thresholds:
            # 预测为正类
            pred_pos = (scores >= threshold).astype(int)
            
            # 计算 TP, FP, TN, FN
            TP = np.sum((pred_pos == 1) & (labels == 1))
            FP = np.sum((pred_pos == 1) & (labels == 0))
            TN = np.sum((pred_pos == 0) & (labels == 0))
            FN = np.sum((pred_pos == 0) & (labels == 1))
            
            # 计算指标
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            accuracies.append(accuracy)
        
        return precisions, recalls, f1s, accuracies
    
    def evaluate_with_ground_truth(self, model, eval_data, ground_truth_df, n_users, n_items, device):
        """
        根据 ground truth 进行评估
        
        Args:
            model: 训练好的模型
            eval_data: 评估数据加载器
            ground_truth_df: 包含 userID 和 itemID 的 DataFrame
            n_users: 用户数量
            n_items: 物品数量
            device: 计算设备
        
        Returns:
            dict: 评估指标
        """
        model.eval()
        
        # 1. 获取评估集中涉及的所有用户ID
        eval_user_ids = []
        with torch.no_grad():
            for batch_data in eval_data:
                user_ids = batch_data[0]  # 用户ID列表
                eval_user_ids.append(user_ids)
        eval_user_ids = torch.cat(eval_user_ids, dim=0).cpu().numpy()
        
        # 2. 获取所有用户对所有物品的预测分数
        all_scores = []
        with torch.no_grad():
            # 需要为所有用户计算分数
            # 创建一个包含所有用户的 batch
            all_user_ids = torch.arange(n_users, device=device)
            # 分批处理以避免内存问题
            batch_size = 64
            for i in range(0, n_users, batch_size):
                batch_user_ids = all_user_ids[i:min(i+batch_size, n_users)]
                scores = model.full_sort_predict([batch_user_ids])
                all_scores.append(scores.cpu())
        
        # 合并为完整的 [n_users, n_items] 分数矩阵
        scores_matrix = torch.cat(all_scores, dim=0)
        
        # 3. 构建标签矩阵（基于 ground_truth_df）
        labels_matrix = torch.zeros(n_users, n_items)
        for _, row in ground_truth_df.iterrows():
            user_id = int(row['userID'])
            item_id = int(row['itemID'])
            if 0 <= user_id < n_users and 0 <= item_id < n_items:
                labels_matrix[user_id, item_id] = 1.0
        
        # 4. 进行评估
        return self.evaluate(scores_matrix, labels_matrix)
    
    def get_curve_data_gmnn_style(self, scores_matrix, labels_matrix):
        """
        获取曲线数据，使用 sklearn 确保与 AUC 计算一致

        Args:
            scores_matrix: [n_users, n_items] 预测分数矩阵
            labels_matrix: [n_users, n_items] 标签矩阵 (0/1)

        Returns:
            dict: 包含 tpr, fpr, precision, recall 数组
        """
        # 展平为 1D 数组
        scores = scores_matrix.flatten().cpu().numpy()
        labels = labels_matrix.flatten().cpu().numpy()

        # 使用 sklearn 计算 ROC 曲线（确保与 roc_auc_score 一致）
        fpr, tpr, _ = roc_curve(labels, scores)

        # 使用 sklearn 计算 PR 曲线（确保与 average_precision_score 一致）
        precision, recall, _ = precision_recall_curve(labels, scores)

        return {
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'recall': recall
        }
    
    def evaluate_disease_column(self, pred_col, label_col):
        """
        按疾病列评估（参考 RWR_denovo.py 的评估方式）
        对单个疾病的预测结果进行评估
        
        Args:
            pred_col: [n_tsRNAs] 该疾病列的预测分数
            label_col: [n_tsRNAs] 该疾病列的真实标签 (0/1)
        
        Returns:
            dict: {'tpr': array, 'fpr': array, 'precision': array, 'recall': array,
                   'accuracy': array, 'auc': float, 'aupr': float}
        """
        pred_col = np.array(pred_col)
        label_col = np.array(label_col)
        
        # 按预测分数降序排序
        sort_index = np.argsort(-pred_col)
        sorted_labels = label_col[sort_index]
        
        n_samples = len(sorted_labels)
        
        tpr_list = []
        fpr_list = []
        precision_list = []
        recall_list = []
        accuracy_list = []
        
        # 遍历所有 cutoff 点（从 1 到 n_samples，参考 RWR_denovo.py）
        for cutoff in range(1, n_samples + 1):
            # 预测为正类的部分
            P_vector = sorted_labels[0:cutoff]
            # 预测为负类的部分
            N_vector = sorted_labels[cutoff:]
            
            # 计算 TP, FP, TN, FN
            TP = np.sum(P_vector == 1)
            FP = np.sum(P_vector == 0)
            TN = np.sum(N_vector == 0)
            FN = np.sum(N_vector == 1)
            
            # 计算指标
            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = tpr  # recall = TPR
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
        
        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)
        precision_arr = np.array(precision_list)
        recall_arr = np.array(recall_list)
        accuracy_arr = np.array(accuracy_list)
        
        # 计算 AUC 和 AUPR
        auc = np.trapz(tpr_arr, fpr_arr)
        aupr = np.trapz(precision_arr, recall_arr)
        
        return {
            'tpr': tpr_arr, 
            'fpr': fpr_arr,
            'precision': precision_arr, 
            'recall': recall_arr,
            'accuracy': accuracy_arr,
            'auc': auc, 
            'aupr': aupr
        }
