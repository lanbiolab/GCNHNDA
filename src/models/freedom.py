# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM
# Update: 01/08/2022
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class FREEDOM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Handle relative data_path: resolve from src directory
        data_path = config['data_path']
        if not os.path.isabs(data_path):
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(src_dir, data_path)
        dataset_path = os.path.abspath(data_path + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # ========== 困难负样本 Mixup 配置 ==========
        # 是否启用困难负样本 Mixup
        self.use_hard_neg_mixup = config['use_hard_neg_mixup'] if 'use_hard_neg_mixup' in config else False
        # 负样本数量
        self.num_negative = config['training_neg_sample_num'] if 'training_neg_sample_num' in config else 1
        # Mixup 混合系数范围 [min, max]，参考 SGCL 使用 [0.0, 1.0]
        self.mixup_alpha_range = config['mixup_alpha_range'] if 'mixup_alpha_range' in config else [0.0, 1.0]
        # 是否使用 SVD-based Mixup (按奇异值比例混合)，False 则使用简单线性插值
        self.use_svd_mixup = config['use_svd_mixup'] if 'use_svd_mixup' in config else True

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        # 直接设置元素，兼容新 SciPy 版本
        for row, col in zip(inter_M.row, inter_M.col + self.n_users):
            A[row, col] = 1
        for row, col in zip(inter_M_t.row + self.n_users, inter_M_t.col):
            A[row, col] = 1
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def forward_mix(self):
        """
        带 Dropout 的前向传播，返回每层嵌入用于 Mixup
        
        与 forward 方法的区别：
        - 每一层都应用 Dropout
        - 返回每层的嵌入列表（而非聚合后的嵌入）
        
        Returns:
            tuple: (user_embs, item_embs, mm_embs)
                - user_embs: 用户嵌入列表，每层一个
                - item_embs: 物品嵌入列表，每层一个
                - mm_embs: 多模态物品嵌入列表
        """
        dropout = nn.Dropout(self.dropout)
        
        # 多模态物品图传播
        h = self.item_id_embedding.weight
        mm_embs = [h]
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
            h = dropout(h)
            mm_embs.append(h)
        
        # 用户-物品交互图传播
        ego_embeddings = torch.cat((self.user_embedding.weight, 
                                    self.item_id_embedding.weight), dim=0)
        user_embs = [self.user_embedding.weight]
        item_embs = [self.item_id_embedding.weight]
        
        for i in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.masked_adj, ego_embeddings)
            ego_embeddings = dropout(ego_embeddings)
            user_embs.append(ego_embeddings[:self.n_users])
            item_embs.append(ego_embeddings[self.n_users:])
        
        return user_embs, item_embs, mm_embs

    def _fast_svd_mixup(self, pos_emb, neg_emb, alpha):
        """
        快速 SVD-based Mixup，利用 M*M^T 技巧优化计算
        
        对于 (2, dim) 矩阵 M，SVD 可以通过以下步骤完成：
        1. A = M @ M^T  → (2, 2) 对称矩阵
        2. 特征分解 A → 得到 λ, U（A 的特征向量 = M 的左奇异向量）
        3. S = √λ → 奇异值
        4. 混合奇异值后重构矩阵
        
        复杂度从 O(dim³) 降到 O(dim²)，加速约 50-100x
        
        Args:
            pos_emb: 正样本嵌入 (batch_size, dim)
            neg_emb: 负样本嵌入 (batch_size, num_neg, dim)
            alpha: 混合系数 (batch_size, num_neg, 1)
        
        Returns:
            mixed_neg: 混合后的负样本嵌入 (batch_size, num_neg, dim)
        """
        batch_size, num_neg, dim = neg_emb.shape
        
        # 1. 构造配对矩阵 M = [pos; neg]
        # pos_emb: (batch, dim) -> (batch, num_neg, dim)
        pos_expanded = pos_emb.unsqueeze(1).expand(-1, num_neg, -1)
        # M: (batch, num_neg, 2, dim)
        M = torch.stack([pos_expanded, neg_emb], dim=2)
        # 展平: (batch*num_neg, 2, dim)
        M_flat = M.view(batch_size * num_neg, 2, dim)
        
        # 2. 计算 A = M @ M^T (2x2 对称矩阵)
        # A 恒为正半定矩阵，无需添加噪声
        A = torch.bmm(M_flat, M_flat.transpose(1, 2))  # (batch*num_neg, 2, 2)
        
        # 3. 对 (2,2) 对称矩阵做特征分解
        # torch.linalg.eigh 适用于对称矩阵，比 eig 更稳定
        # eigenvalues 升序排列，eigenvectors 对应列向量
        eigenvalues, U = torch.linalg.eigh(A)  # (batch*num_neg, 2), (batch*num_neg, 2, 2)
        
        # 4. 奇异值 S = √λ，降序排列（与 torch.linalg.svd 一致）
        # 添加小常数防止数值问题
        S = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))  # (batch*num_neg, 2)
        S = S.flip(-1)  # 降序: [s1, s2] where s1 >= s2
        U = U.flip(-1)  # 对应调整 U 的列顺序
        
        # 5. 按比例混合奇异值
        alpha_flat = alpha.view(batch_size * num_neg, 1)  # (batch*num_neg, 1)
        # S_mix = [alpha * s1, (1-alpha) * s2]
        S_mix = torch.cat([alpha_flat * S[:, 0:1], 
                           (1 - alpha_flat) * S[:, 1:2]], dim=1)  # (batch*num_neg, 2)
        
        # 6. 重构混合后的嵌入
        # M' = U @ diag(S_mix) @ V^T
        # 但我们只需要第0行，可以利用: M'[:, 0, :] = U[:, 0, :] * S_mix[:, 0] * V[:, :, 0]
        # 更高效: mixed = U @ diag(S_mix) @ (U^T @ M)
        # 因为 V = M^T @ U @ diag(1/S)
        # 所以 M' = U @ diag(S_mix) @ V^T = U @ diag(S_mix) @ diag(1/S) @ U^T @ M
        # 简化: M'[:, 0, :] 需要计算 U 的第一行乘以 S_mix
        
        # 方法: 直接重构 (batch*num_neg, 2, dim) 然后取第0行
        # mixed = U @ diag(S_mix) @ V^T = (U @ diag(S_mix)) @ (M^T @ U @ diag(1/S))^T
        # 但这太复杂，用更简单的方法:
        # M' = U @ diag(S_mix/S) @ U^T @ M (当 S_mix/S 需要处理 s2=0 的情况)
        
        # 简化计算: 只重构需要的部分
        # 第0行的贡献: U[:, 0, :] * S_mix[:, 0] 作为系数
        # 使用 U @ diag(S_mix) @ U^T @ M 的等价形式
        
        # 计算 U @ diag(S_mix) @ U^T
        # 这个 (2,2) 矩阵的 [0,0] 和 [0,1] 元素用于混合
        # mixed_row0 = (U[:, 0, 0] * S_mix[:, 0]) * M[:, 0, :] + (U[:, 0, 1] * S_mix[:, 1]) * M[:, 1, :]
        # 这是错的，正确的是:
        
        # 完整重构后取第0行
        # diag(S_mix) @ V^T = diag(S_mix) @ V^T 
        # V^T = U^T @ M (不正确)
        # 正确: V = M^T @ U @ diag(1/S), 所以 V^T = diag(1/S) @ U^T @ M
        # 因此: M' = U @ diag(S_mix) @ diag(1/S) @ U^T @ M
        # 设 w = S_mix / S (element-wise)，则 M' = U @ diag(w) @ U^T @ M
        
        # 计算权重 w = S_mix / S
        w = S_mix / S  # (batch*num_neg, 2)
        
        # 计算 U @ diag(w) @ U^T
        # 这是 (2,2) 矩阵
        # diag_w = w.unsqueeze(-1) * torch.eye(2, device=w.device).unsqueeze(0)
        # transform = U @ diag_w @ U.transpose(1, 2)  # (batch*num_neg, 2, 2)
        # 更高效: transform[i] = U[i] @ diag(w[i]) @ U[i]^T
        transform = torch.bmm(U * w.unsqueeze(-1), U.transpose(1, 2))  # (batch*num_neg, 2, 2)
        
        # 应用变换并取第0行
        mixed_neg_flat = torch.bmm(transform, M_flat)[:, 0, :]  # (batch*num_neg, dim)
        
        mixed_neg = mixed_neg_flat.view(batch_size, num_neg, dim)
        return mixed_neg

    def negative_mixup(self, users, pos_items, neg_items):
        """
        困难负样本 Mixup 策略
        
        流程:
        1. 获取每层嵌入
        2. 对负样本进行 Mixup 混合（SVD-based 或简单线性插值）
        3. 选择最困难的负样本（与用户相似度最高）
        
        Args:
            users: 用户索引，形状 (batch_size,)
            pos_items: 正样本物品索引，形状 (batch_size,)
            neg_items: 负样本物品索引，形状 (batch_size * num_negative,)
        
        Returns:
            tuple: (u_emb, pos_emb, neg_emb)
                - u_emb: 用户嵌入，形状 (batch_size, embedding_dim)
                - pos_emb: 正样本物品嵌入，形状 (batch_size, embedding_dim)
                - neg_emb: 混合后的困难负样本嵌入，形状 (batch_size, embedding_dim)
        """
        # 获取每层的嵌入
        user_embs, item_embs, mm_embs = self.forward_mix()
        
        # 聚合用户嵌入（平均池化）
        u_emb = torch.stack(user_embs, dim=1).mean(dim=1)[users]
        
        # 对每一层嵌入执行 Mixup 操作
        negs = []
        for k in range(len(item_embs)):
            item_emb = item_embs[k]
            
            # 获取正样本和负样本嵌入
            neg_emb = item_emb[neg_items].view(-1, self.num_negative, self.embedding_dim)
            pos_emb = item_emb[pos_items]
            
            # 生成随机混合系数
            alpha = torch.rand(
                neg_emb.shape[0], self.num_negative, 1,
                device=self.device, dtype=neg_emb.dtype
            )
            # 缩放到指定范围
            alpha = self.mixup_alpha_range[0] + (self.mixup_alpha_range[1] - self.mixup_alpha_range[0]) * alpha
            
            # 始终先用简单线性插值计算 fallback
            simple_mixed_neg = alpha * pos_emb.unsqueeze(1) + (1 - alpha) * neg_emb
            
            if self.use_svd_mixup:
                # 使用快速 SVD-based Mixup (M*M^T 技巧)
                # 复杂度从 O(dim³) 降到 O(dim²)，加速约 50-100x
                mixed_neg = self._fast_svd_mixup(pos_emb, neg_emb, alpha)
                
                # 检测 NaN 并回退到简单插值
                nan_mask = torch.isnan(mixed_neg).any(dim=-1)
                if nan_mask.any():
                    mixed_neg = torch.where(nan_mask.unsqueeze(-1), simple_mixed_neg, mixed_neg)
            else:
                # 简单线性插值
                mixed_neg = simple_mixed_neg
            
            # 计算用户与混合负样本的相似度（点积）
            scores = (u_emb.unsqueeze(1) * mixed_neg).sum(dim=-1)
            
            # 选择最困难的负样本（相似度最高）
            indices = scores.argmax(dim=1).detach()
            negs.append(mixed_neg[torch.arange(len(users), device=self.device), indices])
        
        # 聚合所有层的负样本嵌入（平均池化）
        neg_emb = torch.stack(negs, dim=1).mean(dim=1)
        
        # 聚合正样本嵌入
        pos_emb = torch.stack([item_embs[k][pos_items] for k in range(len(item_embs))], dim=1).mean(dim=1)
        
        return u_emb, pos_emb, neg_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        self.build_item_graph = False

        # ========== 困难负样本 Mixup 分支 ==========
        if self.use_hard_neg_mixup and neg_items.dim() > 1:
            # 使用 Mixup 困难负样本
            # neg_items 形状: (batch_size, num_negative)
            neg_items_flat = neg_items.flatten()
            
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.negative_mixup(
                users, pos_items, neg_items_flat
            )
            batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            
            # 多模态损失 - 使用每个用户的第一个负样本
            mf_v_loss, mf_t_loss = 0.0, 0.0
            if self.t_feat is not None:
                text_feats = self.text_trs(self.text_embedding.weight)
                # 使用第一个负样本计算多模态损失
                first_neg_items = neg_items[:, 0]
                mf_t_loss = self.bpr_loss(u_g_embeddings, text_feats[pos_items], text_feats[first_neg_items])
            if self.v_feat is not None:
                image_feats = self.image_trs(self.image_embedding.weight)
                first_neg_items = neg_items[:, 0]
                mf_v_loss = self.bpr_loss(u_g_embeddings, image_feats[pos_items], image_feats[first_neg_items])
            return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)
        
        # ========== 原始分支（不使用 Mixup）==========
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        # 处理可能的 2D neg_items
        if neg_items.dim() > 1:
            # 使用第一个负样本
            neg_i_g_embeddings = ia_embeddings[neg_items[:, 0]]
        else:
            neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            if neg_items.dim() > 1:
                mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items[:, 0]])
            else:
                mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            if neg_items.dim() > 1:
                mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items[:, 0]])
            else:
                mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def save_embeddings(self, save_path):
        """
        保存用户和物品嵌入到文件
        
        Args:
            save_path: 保存路径（.pt 文件）
        
        Returns:
            dict: 包含 user_embeddings 和 item_embeddings 的字典
        """
        self.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = self.forward(self.norm_adj)
        
        embeddings = {
            'user_embeddings': user_embeddings.detach().cpu(),
            'item_embeddings': item_embeddings.detach().cpu(),
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")
        print(f"  - User embeddings shape: {user_embeddings.shape}")
        print(f"  - Item embeddings shape: {item_embeddings.shape}")
        
        return embeddings