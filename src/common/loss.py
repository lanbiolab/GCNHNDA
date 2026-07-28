# coding: utf-8
# @email  : enoche.chow@gmail.com


import torch
import torch.nn as nn


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss


class InfoNCELoss(nn.Module):
    r"""InfoNCE Loss for Contrastive Learning
    
    对比学习的 InfoNCE 损失函数，通过最大化同一节点在不同增强视图上表示的相似度，
    最小化不同节点表示的相似度。
    
    数学公式：
        L_InfoNCE = -∑ log [exp(sim(z_i, z_i^+)/τ) / ∑_k exp(sim(z_i, z_k)/τ)]
    
    Args:
        temp (float): 温度参数 τ，控制分布的平滑度。默认 0.2
    
    Shape:
        - view1: (N, D) 第一个视图的节点表示
        - view2: (N, D) 第二个视图的节点表示
        - Output: scalar
    
    Examples::
        >>> loss = InfoNCELoss(temp=0.2)
        >>> view1 = torch.randn(100, 128)
        >>> view2 = torch.randn(100, 128)
        >>> output = loss(view1, view2)
    """
    def __init__(self, temp=0.2):
        super(InfoNCELoss, self).__init__()
        self.temp = temp
    
    def forward(self, view1, view2):
        # L2 归一化
        view1 = nn.functional.normalize(view1, dim=1)
        view2 = nn.functional.normalize(view2, dim=1)
        
        # 正样本对相似度 (对角线)
        pos_score = (view1 * view2).sum(dim=1)
        
        # 全局相似度矩阵
        ttl_score = torch.matmul(view1, view2.T)
        
        # InfoNCE 损失
        # log(exp(pos/τ) / sum(exp(all/τ))) = pos/τ - log(sum(exp(all/τ)))
        pos_score = torch.exp(pos_score / self.temp)
        ttl_score = torch.exp(ttl_score / self.temp).sum(dim=1)
        
        loss = -torch.log(pos_score / ttl_score).mean()
        return loss