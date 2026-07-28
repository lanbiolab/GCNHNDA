# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
MLP Predictor
基于 FREEDOM 第一阶段嵌入的二分类预测器
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    """
    3 层 MLP 预测器
    
    输入: 用户嵌入和物品嵌入的拼接 (2 * embedding_dim)
    输出: 预测概率 (0-1)
    
    结构 (逐层减半):
        Input: 2 * embedding_dim
        Linear(2*dim -> dim) -> ReLU -> Dropout
        Linear(dim -> dim/2) -> ReLU -> Dropout
        Linear(dim/2 -> 1) -> Sigmoid
    """
    
    def __init__(self, embedding_dim, hidden_dims=None, dropout=0.3):
        """
        Args:
            embedding_dim: 嵌入维度 (来自 FREEDOM)
            hidden_dims: 隐藏层维度列表，默认逐层减半 [embedding_dim, embedding_dim//2]
            dropout: Dropout 概率
        """
        super(MLPPredictor, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 默认逐层减半
        if hidden_dims is None:
            hidden_dims = [embedding_dim, embedding_dim // 2]
        
        # 输入维度为用户和物品嵌入拼接
        input_dim = 2 * embedding_dim
        
        # 构建 3 层 MLP
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # 第一层: input_dim -> hidden_dims[0]
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # 第二层: hidden_dims[0] -> hidden_dims[1]
        self.layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        
        # 输出层: hidden_dims[1] -> 1
        self.output_layer = nn.Linear(hidden_dims[1], 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, user_emb, item_emb):
        """
        前向传播
        
        Args:
            user_emb: 用户嵌入 [batch_size, embedding_dim]
            item_emb: 物品嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 预测概率 [batch_size, 1]
        """
        # 拼接用户和物品嵌入
        x = torch.cat([user_emb, item_emb], dim=-1)  # [batch_size, 2*embedding_dim]
        
        # 通过隐藏层
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 输出层 + Sigmoid
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        
        return x.squeeze(-1)  # [batch_size]
    
    def predict_all_items(self, user_emb, all_item_emb):
        """
        预测用户对所有物品的分数
        
        Args:
            user_emb: 单个用户嵌入 [embedding_dim]
            all_item_emb: 所有物品嵌入 [n_items, embedding_dim]
        
        Returns:
            torch.Tensor: 预测分数 [n_items]
        """
        # 扩展用户嵌入
        user_emb_expanded = user_emb.unsqueeze(0).expand(all_item_emb.size(0), -1)
        
        # 批量预测
        scores = self.forward(user_emb_expanded, all_item_emb)
        
        return scores


class MLPPredictorWithEmbeddings(nn.Module):
    """
    带嵌入查找的 MLP 预测器
    直接使用用户/物品 ID 进行预测
    """
    
    def __init__(self, user_embeddings, item_embeddings, hidden_dims=None, dropout=0.3, freeze_embeddings=True):
        """
        Args:
            user_embeddings: 预训练的用户嵌入 [n_users, embedding_dim]
            item_embeddings: 预训练的物品嵌入 [n_items, embedding_dim]
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 概率
            freeze_embeddings: 是否冻结嵌入层
        """
        super(MLPPredictorWithEmbeddings, self).__init__()
        
        self.n_users = user_embeddings.size(0)
        self.n_items = item_embeddings.size(0)
        self.embedding_dim = user_embeddings.size(1)
        
        # 嵌入层 (使用预训练嵌入)
        self.user_embedding = nn.Embedding.from_pretrained(user_embeddings, freeze=freeze_embeddings)
        self.item_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=freeze_embeddings)
        
        # MLP 预测器
        self.mlp = MLPPredictor(self.embedding_dim, hidden_dims, dropout)
    
    def forward(self, user_ids, item_ids):
        """
        前向传播
        
        Args:
            user_ids: 用户 ID [batch_size]
            item_ids: 物品 ID [batch_size]
        
        Returns:
            torch.Tensor: 预测概率 [batch_size]
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return self.mlp(user_emb, item_emb)
    
    def predict_all_items(self, user_id, device):
        """
        预测单个用户对所有物品的分数
        
        Args:
            user_id: 用户 ID
            device: 计算设备
        
        Returns:
            torch.Tensor: 预测分数 [n_items]
        """
        user_emb = self.user_embedding.weight[user_id]
        all_item_emb = self.item_embedding.weight
        return self.mlp.predict_all_items(user_emb, all_item_emb)
    
    def full_sort_predict(self, interaction):
        """
        兼容原有评估接口
        
        Args:
            interaction: [user_ids] 或 [user_ids, mask_matrix]
        
        Returns:
            torch.Tensor: 所有物品的预测分数
        """
        user_ids = interaction[0]
        
        # 批量处理所有用户
        all_scores = []
        for user_id in user_ids:
            scores = self.predict_all_items(user_id.item() if torch.is_tensor(user_id) else user_id, 
                                           user_ids.device)
            all_scores.append(scores)
        
        return torch.stack(all_scores, dim=0)


def build_mlp_predictor(config, user_embeddings, item_embeddings):
    """
    构建 MLP 预测器
    
    Args:
        config: 配置对象（Config 类）
        user_embeddings: 用户嵌入
        item_embeddings: 物品嵌入
    
    Returns:
        MLPPredictorWithEmbeddings: 预测器模型
    """
    hidden_dims = config['mlp_hidden_dims'] if 'mlp_hidden_dims' in config else None  # 默认逐层减半
    dropout = config['mlp_dropout'] if 'mlp_dropout' in config else 0.3
    freeze_embeddings = config['freeze_embeddings'] if 'freeze_embeddings' in config else True
    
    model = MLPPredictorWithEmbeddings(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        hidden_dims=hidden_dims,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings
    )
    
    return model
