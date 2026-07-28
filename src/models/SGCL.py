import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, scl_loss
from data.augmentor import GraphAugmentor
from util.embedding_loader import PretrainedEmbeddingLoader
from util.pretrained_embedding_manager import PretrainedEmbeddingManager
import os


"""
SGCL (Self-supervised Graph Contrastive Learning) 模型实现。

该模块实现了基于自监督图对比学习的推荐算法，通过图增强和对比学习
增强用户和物品的表示学习，提高推荐系统的性能。

核心组件：
    - 图数据增强：支持节点丢弃（Node Dropout）、边丢弃（Edge Dropout）
      和多层边丢弃（Multi-layer Edge Dropout）三种策略
    - 负样本 Mixup：通过混合正负样本嵌入增强负样本多样性，提高模型鲁棒性
    - 对比学习：通过对比不同增强视图学习更好的表示

核心数学公式：
    
    1. BPR (Bayesian Personalized Ranking) 损失：
        L_BPR = -∑_{(u,i,j)∈D} log σ(p_{ui} - p_{uj})
        其中 p_{ui} 是用户 u 对物品 i 的预测评分，j 是负样本
    
    2. 自监督对比学习（SCL）损失：
        L_SCL = -∑_{i} log (exp(sim(z_i, z_i^+)/τ) / ∑_{k} exp(sim(z_i, z_k)/τ))
        其中 sim(·) 是相似度函数，τ 是温度参数，z_i^+ 是正样本
    
    3. 总损失函数：
        L_total = L_BPR + λ * L2 + β * L_SCL
        其中 λ 是 L2 正则化系数，β 是对比学习损失权重
    
    4. Mixup 操作（负样本增强）：
        emb_neg_mix = α * emb_pos + (1-α) * emb_neg
        其中 α ~ Uniform(0,1) 是随机生成的混合系数

References:
    - SGL: Self-supervised Graph Learning for Recommendation
    - Wu, S., et al. (2021). Self-supervised Graph Learning for Recommendation. CIKM.

Classes:
    SGCL: 自监督图对比学习推荐模型主类
    SGL_Encoder: 图神经网络编码器类
"""


class SGCL(GraphRecommender):
    """自监督图对比学习推荐模型。
    
    该类实现了基于图神经网络和对比学习的推荐算法，通过图数据增强
    和自监督对比学习来增强用户和物品的表示学习，从而提高推荐性能。
    
    主要特点：
        - 支持多种图增强策略（节点丢弃、边丢弃、多层边丢弃）
        - 使用负样本 Mixup 增强负样本多样性
        - 结合 BPR 损失和对比学习损失进行联合优化
    
    继承自 GraphRecommender 基类。
    
    Attributes:
        best_user_emb (torch.Tensor): 最佳用户嵌入，形状为 (num_users, emb_size)
        best_item_emb (torch.Tensor): 最佳物品嵌入，形状为 (num_items, emb_size)
        device (torch.device): 运行设备（CUDA 或 CPU）
        cl_rate (float): 对比学习损失权重（β参数）
        aug_type (int): 图增强类型（0-节点丢弃, 1-边丢弃, 2-多层边丢弃）
        drop_rate (float): Dropout 比例
        n_layers (int): 图神经网络层数
        temp (float): 对比学习温度参数（τ）
        model (SGL_Encoder): 图神经网络编码器实例
    """
    
    def __init__(self, conf, training_set,  test_set):
        """初始化 SGCL 推荐模型。
        
        Args:
            conf (dict): 配置字典，包含模型超参数
            training_set: 训练数据集
            test_set: 测试数据集
        
        Notes:
            从配置中读取的关键参数：
                - cl_rate (β): 对比学习损失权重，控制自监督学习的重要性
                - aug_type: 图增强类型
                    0: 节点丢弃（Node Dropout）
                    1: 边丢弃（Edge Dropout）
                    2: 多层边丢弃（Multi-layer Edge Dropout）
                - drop_rate: 图增强中的 Dropout 比例
                - n_layers: 图神经网络的层数
                - temp (τ): 对比学习中的温度参数，控制分布的平滑度
        """
        super(SGCL, self).__init__(conf, training_set, test_set)
        self.best_user_emb = None
        self.best_item_emb = None
        # 检测设备（CUDA或CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        args = OptionConf(self.config['SGCL'])
        self.cl_rate = float(args['-beta'])  # 对比学习损失权重（β）
        aug_type = self.aug_type = int(args['-augtype'])  # 图增强类型
        drop_rate = float(args['-droprate'])  # Dropout 比例
        n_layers = int(args['-n_layer'])  # GNN 层数
        temp = float(args['-temp'])  # 温度参数（τ）
        num_negative = int(self.config['num.negative'])  # 负采样数量
        
        # 初始化预训练嵌入管理器
        pretrained_manager = None
        try:
            # 获取嵌入文件路径
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tsRNA-origin')
            disease_emb_path = os.path.join(base_dir, 'disease_embeddings_only.npy')
            tsrna_emb_path = os.path.join(base_dir, 'tsrna_embeddings_only.npy')
            id_mapping_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dataset', '5fold_w', 'id_mappings_v2.json')
            
            # 检查文件是否存在
            if os.path.exists(disease_emb_path) and os.path.exists(tsrna_emb_path) and os.path.exists(id_mapping_path):
                print("加载预训练嵌入...")
                loader = PretrainedEmbeddingLoader(disease_emb_path, tsrna_emb_path, device=self.device)
                disease_emb = loader.load_disease_embeddings()
                tsrna_emb = loader.load_tsrna_embeddings()
                
                pretrained_manager = PretrainedEmbeddingManager(
                    self.data, disease_emb, tsrna_emb, 
                    target_emb_size=self.emb_size,
                    id_mapping_path=id_mapping_path,
                    device=self.device
                )
                print(f"预训练嵌入管理器初始化成功。用户数: {len(pretrained_manager.user_id_map)}, 物品数: {len(pretrained_manager.item_id_map)}")
            else:
                print("预训练嵌入文件未找到，使用随机初始化")
        except Exception as e:
            print(f"加载预训练嵌入失败: {e}，使用随机初始化")
        
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, num_negative, aug_type, self.device, pretrained_manager)
        
        # 保存预训练管理器引用，用于访问降维层
        self.pretrained_manager = pretrained_manager

    def train(self):
        """训练模型一个完整的 epoch 周期。
        
        训练流程包括：
        1. 图数据增强：生成两个不同的增强视图
        2. 批次训练：遍历所有训练批次
        3. 负样本 Mixup：增强负样本多样性
        4. 损失计算：计算 BPR 推荐损失、L2 正则化损失和对比学习损失
        5. 反向传播：更新模型参数
        6. 评估：从第 5 个 epoch 开始进行快速评估
        
        损失函数：
            L_total = L_BPR + λ * L2 + β * L_SCL
            其中：
            - L_BPR: Bayesian Personalized Ranking 损失
            - L2: L2 正则化损失
            - L_SCL: 自监督对比学习损失
        
        Notes:
            - 使用 Adam 优化器
            - 包含预训练嵌入的降维层参数
            - 每 100 个 batch 打印一次训练损失
            - 保存验证集上表现最佳的嵌入
        """
        model = self.model.to(self.device)
        
        # 收集所有可训练参数：模型参数 + 预训练嵌入的降维层参数
        params = list(model.parameters())
        if self.pretrained_manager is not None:
            params.extend(self.pretrained_manager.get_projection_parameters())
        
        optimizer = torch.optim.Adam(params, lr=self.lRate)
        print(f"优化器初始化完成，总参数数量: {sum(p.numel() for p in params)}")
        
        for epoch in range(self.maxEpoch):
            # 图数据增强：生成两个不同的增强视图
            # 这些增强视图用于对比学习
            dropped_adj1 = model.graph_reconstruction()  # 第一个增强视图
            dropped_adj2 = model.graph_reconstruction()  # 第二个增强视图
            
            # 批次训练
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, self.num_negative)):
                user_idx, pos_idx, neg_idx = batch
                
                # 负样本 Mixup：增强负样本多样性
                # 返回用户嵌入、正样本物品嵌入和混合后的负样本嵌入
                user_emb, pos_item_emb, neg_item_emb = model.negative_mixup(user_idx, pos_idx, neg_idx)
                
                # 计算 BPR 推荐损失
                # L_BPR = -∑ log σ(p_{ui} - p_{uj})
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                # 计算对比学习损失
                # L_SCL = -∑ log (exp(sim(z_i, z_i^+)/τ) / ∑ exp(sim(z_i, z_k)/τ))
                scl_loss_ = self.cl_rate * model.cal_scl_loss([user_idx, pos_idx, neg_idx], dropped_adj1, dropped_adj2)
                
                # 计算总损失：L_total = L_BPR + λ * L2 + β * L_SCL
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,
                                                    neg_item_emb) + scl_loss_
                
                # 反向传播和优化
                optimizer.zero_grad()  # 清空梯度
                batch_loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数
                
                # 每 100 个 batch 打印一次训练损失
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
            
            # 评估当前模型
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch >= 5:
                self.fast_evaluation(epoch)
        
        # 使用最佳模型嵌入
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        """保存当前模型的最佳用户和物品嵌入，以及降维层状态。
        
        该方法在训练完成后调用，保存模型在验证集上表现最佳时的
        用户和物品嵌入，以及预训练嵌入的降维层状态。
        
        Notes:
            - 使用 torch.no_grad() 上下文，不计算梯度
            - 将最佳嵌入存储在 self.best_user_emb 和 self.best_item_emb 中
            - 如果存在预训练嵌入管理器，则保存降维层状态到文件
        """
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
        
        # 保存降维层状态
        if self.pretrained_manager is not None:
            projection_state = self.pretrained_manager.get_projection_state_dict()
            if projection_state:
                import os
                import json
                from datetime import datetime
                
                # 创建保存目录
                save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_projections')
                os.makedirs(save_dir, exist_ok=True)
                
                # 生成文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_dir, f'projection_layers_{timestamp}.pt')
                
                torch.save(projection_state, save_path)
                print(f"降维层状态已保存到: {save_path}")
                
                # 同时保存元信息
                meta_info = {
                    'timestamp': timestamp,
                    'user_num': self.data.user_num,
                    'item_num': self.data.item_num,
                    'emb_size': self.emb_size,
                    'num_users_with_pretrained': len(self.pretrained_manager.user_id_map),
                    'num_items_with_pretrained': len(self.pretrained_manager.item_id_map)
                }
                meta_path = os.path.join(save_dir, f'projection_layers_{timestamp}.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta_info, f, indent=2)
                print(f"降维层元信息已保存到: {meta_path}")

    def predict(self, u):
        """预测用户对所有物品的偏好分数。
        
        Args:
            u (str or int): 用户的外部 ID
        
        Returns:
            numpy.ndarray: 预测分数数组，形状为 (num_items,)，
                          表示用户 u 对所有物品的预测评分
        
        Notes:
            - 使用点积计算用户嵌入和物品嵌入的相似度
            - 公式：score(u, i) = emb_u · emb_i
            - 结果返回为 numpy 数组，便于后续处理
        """
        u = self.data.get_user_id(u)

        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    """图神经网络编码器，用于学习用户和物品的嵌入表示。
    
    该编码器实现了多层图卷积网络（GCN），支持图数据增强和自监督
    对比学习。通过在不同增强视图上进行前向传播，学习鲁棒的节点表示。
    
    主要功能：
        - 多层图卷积：通过堆叠图卷积层聚合邻居信息
        - 图数据增强：支持节点丢弃和边丢弃策略
        - 负样本 Mixup：增强负样本的多样性
        - 对比学习：计算不同视图间的对比损失
    
    Attributes:
        data: 数据集对象，包含交互图、用户数、物品数等信息
        drop_rate (float): 图增强中的 Dropout 比例
        emb_size (int): 嵌入维度
        n_layers (int): 图神经网络的层数
        temp (float): 对比学习温度参数（τ）
        aug_type (int): 图增强类型（0-节点丢弃, 1-边丢弃, 2-多层边丢弃）
        norm_adj (scipy.sparse): 归一化的邻接矩阵
        num_negative (int): 负采样数量
        device (torch.device): 运行设备
        embedding_dict (nn.ParameterDict): 用户和物品的嵌入参数
        sparse_norm_adj (torch.sparse.FloatTensor): 稀疏格式的归一化邻接矩阵
        pretrained_manager: 预训练嵌入管理器，用于初始化嵌入
    """
    
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, num_negative, aug_type, device, pretrained_manager=None):
        """初始化图神经网络编码器。
        
        Args:
            data: 数据集对象，包含用户-物品交互图
            emb_size (int): 嵌入向量的维度
            drop_rate (float): 图增强中的 Dropout 比例
            n_layers (int): 图神经网络的层数
            temp (float): 对比学习温度参数（τ）
            num_negative (int): 负采样数量
            aug_type (int): 图增强类型
                0: 节点丢弃（Node Dropout）
                1: 边丢弃（Edge Dropout）
                2: 多层边丢弃（Multi-layer Edge Dropout）
            device (torch.device): 运行设备（CUDA 或 CPU）
            pretrained_manager: 预训练嵌入管理器，用于初始化嵌入（可选）
        
        Notes:
            - 如果提供 pretrained_manager，则使用融合初始化策略
            - 否则使用 Xavier 均匀初始化
            - 将归一化邻接矩阵转换为稀疏张量格式
        """
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.num_negative = num_negative
        self.device = device
        self.pretrained_manager = pretrained_manager
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(self.device)

    def _init_model(self):
        """初始化用户和物品的嵌入参数。
        
        如果提供了预训练嵌入管理器，则使用融合初始化策略（alpha * random + (1-alpha) * pretrained），
        否则使用 Xavier 均匀初始化。
        
        Returns:
            nn.ParameterDict: 包含用户和物品嵌入的参数字典
                - 'user_emb': 用户嵌入矩阵，形状为 (user_num, emb_size)
                - 'item_emb': 物品嵌入矩阵，形状为 (item_num, emb_size)
        
        Notes:
            - 融合初始化：alpha * random_emb + (1-alpha) * pretrained_emb
            - Xavier 均匀初始化公式：
                U ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
        """
        if self.pretrained_manager is not None:
            # 使用融合初始化
            print("使用预训练嵌入进行融合初始化...")
            user_emb, item_emb = self.pretrained_manager.get_fusion_embeddings(alpha=0.5)
            embedding_dict = nn.ParameterDict({
                'user_emb': nn.Parameter(user_emb),
                'item_emb': nn.Parameter(item_emb),
            })
        else:
            # 使用 Xavier 均匀初始化
            initializer = nn.init.xavier_uniform_
            embedding_dict = nn.ParameterDict({
                'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
                'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
            })
        return embedding_dict

    def graph_reconstruction(self):
        """生成增强的图邻接矩阵。
        
        根据配置的增强类型，对原始用户-物品交互图进行增强，
        生成一个或多个增强视图用于对比学习。
        
        Returns:
            torch.sparse.FloatTensor 或 list: 增强的邻接矩阵
                - 如果 aug_type == 0 或 1：返回单个增强的邻接矩阵
                - 如果 aug_type == 2：返回包含 n_layers 个增强邻接矩阵的列表
        
        Notes:
            增强类型说明：
                - aug_type == 0：节点丢弃（Node Dropout），随机丢弃部分节点
                - aug_type == 1：边丢弃（Edge Dropout），随机丢弃部分边
                - aug_type == 2：多层边丢弃，为每一层生成不同的增强视图
        """
        if self.aug_type == 0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        """执行随机图增强操作。
        
        根据配置的增强类型，对原始交互图应用不同的增强策略，
        然后将结果转换为拉普拉斯矩阵格式。
        
        Returns:
            torch.sparse.FloatTensor: 增强后的拉普拉斯矩阵（稀疏张量格式）
        
        Notes:
            增强策略：
                1. 节点丢弃（Node Dropout）：
                   随机丢弃一部分节点及其所有连接边
                2. 边丢弃（Edge Dropout）：
                   随机丢弃一部分边（用户-物品交互）
            
            增强后的图需要转换为拉普拉斯矩阵，用于图卷积操作：
                L = D^(-1/2) * A * D^(-1/2)
                其中 D 是度矩阵，A 是邻接矩阵
        """
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).to(self.device)

    def forward(self, perturbed_adj=None):
        """执行多层图卷积，计算用户和物品的嵌入表示。
        
        通过堆叠图卷积层，聚合用户和物品的邻居信息，学习高阶的
        图结构特征。支持使用原始图或增强图进行前向传播。
        
        Args:
            perturbed_adj (torch.sparse.FloatTensor or list, optional):
                增强的邻接矩阵。如果为 None，使用原始邻接矩阵。
                如果是列表，则每层使用不同的增强视图。
        
        Returns:
            tuple: (user_all_embeddings, item_all_embeddings)
                - user_all_embeddings (torch.Tensor): 用户嵌入，形状为 (user_num, emb_size)
                - item_all_embeddings (torch.Tensor): 物品嵌入，形状为 (item_num, emb_size)
        
        Notes:
            图卷积操作：
                H^(l+1) = σ(A * H^(l))
                其中 A 是归一化邻接矩阵，H^(l) 是第 l 层的节点表示
            
            嵌入聚合：
                使用平均池化聚合所有层的嵌入：
                H_final = mean([H^(0), H^(1), ..., H^(L)])
        """
        # 拼接用户和物品的初始嵌入
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]  # 收集每一层的嵌入
        
        # 执行多层图卷积
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                # 使用增强图
                if isinstance(perturbed_adj, list):
                    # 多层增强：每一层使用不同的增强视图
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    # 单层增强：所有层使用同一个增强视图
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                # 使用原始图
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # 聚合所有层的嵌入（平均池化）
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # 分离用户和物品嵌入
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def forward_mix(self):
        """执行带 Dropout 的多层图卷积，用于负样本 Mixup。
        
        与 forward 方法类似，但在每一层都应用 Dropout，以增强负样本的
        多样性。该方法专门用于生成混合负样本的嵌入表示。
        
        Returns:
            tuple: (user_embs, item_embs)
                - user_embs (torch.Tensor): 用户嵌入，形状为 (user_num, emb_size)
                - item_embs (torch.Tensor): 物品嵌入，形状为 (item_num, emb_size)
        
        Notes:
            与 forward 方法的主要区别：
                - 每一层都应用 Dropout（丢弃率 10%）
                - 不支持使用增强图
                - 专门用于负样本 Mixup 操作
        """
        self.dropout = nn.Dropout(0.1)  # 10% Dropout 率
        
        # 拼接用户和物品的初始嵌入
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        
        # 收集每一层的用户和物品嵌入
        user_embs = [self.embedding_dict['user_emb']]
        item_embs = [self.embedding_dict['item_emb']]
        
        # adj = self._sparse_dropout(self.sparse_norm_adj, 0.5)  # 注释掉的代码
        
        # 执行带 Dropout 的多层图卷积
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddings = self.dropout(ego_embeddings)  # 应用 Dropout
            user_embs.append(ego_embeddings[:self.data.user_num])
            item_embs.append(ego_embeddings[self.data.user_num:])
        
        # 聚合所有层的用户嵌入（平均池化）
        user_embs = torch.stack(user_embs, dim=1)
        user_embs = torch.mean(user_embs, dim=1)
        return user_embs, item_embs

    def negative_mixup(self, user, pos_item, neg_item):
        """通过 Mixup 策略生成增强的负样本嵌入。
        
        该方法通过混合正样本和负样本的嵌入，生成更丰富的负样本表示，
        提高模型的鲁棒性和泛化能力。
        
        Args:
            user (torch.Tensor): 用户索引，形状为 (batch_size,)
            pos_item (torch.Tensor): 正样本物品索引，形状为 (batch_size,)
            neg_item (torch.Tensor): 负样本物品索引，形状为 (batch_size, num_negative)
        
        Returns:
            tuple: (u_emb, pos_item_emb, negs)
                - u_emb (torch.Tensor): 用户嵌入，形状为 (batch_size, emb_size)
                - pos_item_emb (torch.Tensor): 正样本物品嵌入，形状为 (batch_size, emb_size)
                - negs (torch.Tensor): 混合后的负样本嵌入，形状为 (batch_size, emb_size)
        
        Notes:
            Mixup 策略的核心思想：
            通过线性混合正样本和负样本的嵌入，生成新的负样本表示，
            从而增加负样本的多样性，使模型学习更鲁棒的特征表示。
            
            数学公式：
                1. Mixup 操作：
                   emb_mix = α * emb_pos + (1-α) * emb_neg
                   其中 α ~ Uniform(0,1) 是随机生成的混合系数
                
                2. 相似度计算：
                   score(u, i) = u · i （点积）
                
                3. 负样本选择：
                   选择与用户最相似的混合负样本：
                   i* = argmax_i score(u, emb_mix_i)
            
            步骤说明：
                1. 对每一层嵌入都执行 Mixup 操作
                2. 为每个负样本生成随机混合系数
                3. 计算用户与混合负样本的相似度
                4. 选择最相似的负样本
                5. 聚合所有层的负样本嵌入
        """
        # 获取用户和物品的嵌入（带 Dropout）
        user_emb, item_emb = self.forward_mix()
        u_emb = user_emb[user]
        negs = []
        
        # 对每一层嵌入都执行 Mixup 操作
        for k in range(self.n_layers + 1):
            # 获取当前层的正样本和负样本嵌入
            neg_emb = item_emb[k][neg_item]
            pos_emb = item_emb[k][pos_item]
            
            # 重塑负样本嵌入：形状从 (batch_size * num_negative, emb_size)
            # 变为 (batch_size, num_negative, emb_size)
            neg_emb = neg_emb.reshape(-1, self.num_negative, self.emb_size)
            
            # Mixup 操作：生成随机混合系数并混合正负样本
            # 公式：emb_mix = α * emb_pos + (1-α) * emb_neg
            # 其中 α ~ Uniform(0,1)
            alpha = torch.rand_like(neg_emb).to(self.device)
            neg_emb = alpha * pos_emb.unsqueeze(dim=1) + (1 - alpha) * neg_emb
            
            # 计算用户与混合负样本的相似度（点积）
            # 公式：score(u, i) = u · i
            scores = (u_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
            
            # 选择与用户最相似的负样本
            # detach() 防止梯度传播到选择过程
            indices = torch.max(scores, dim=1)[1].detach()
            chosen_neg_emb = neg_emb[torch.arange(neg_emb.size(0)), indices]
            negs.append(chosen_neg_emb)
        
        # 聚合所有层的物品嵌入（平均池化）
        item_emb = torch.stack(item_emb, dim=1)
        item_emb = torch.mean(item_emb, dim=1)
        
        # 聚合所有层的负样本嵌入（平均池化）
        negs = torch.stack(negs, dim=1)
        negs = torch.mean(negs, dim=1)
        
        return u_emb, item_emb[pos_item], negs

    def cal_scl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        """计算自监督对比学习损失。
        
        通过对比两个不同增强视图上的节点表示，学习鲁棒的特征表示。
        该方法实现了 InfoNCE (Noise Contrastive Estimation) 损失函数。
        
        Args:
            idx (list): 包含用户和物品索引的列表，格式为 [user_indices, item_indices]
            perturbed_mat1 (torch.sparse.FloatTensor): 第一个增强视图的邻接矩阵
            perturbed_mat2 (torch.sparse.FloatTensor): 第二个增强视图的邻接矩阵
        
        Returns:
            torch.Tensor: 对比学习损失值（标量）
        
        Notes:
            对比学习的核心思想：
            通过最大化同一节点在不同增强视图上表示的相似度，
            最小化不同节点表示的相似度，从而学习判别性的特征表示。
            
            数学公式（InfoNCE 损失）：
                L_SCL = -∑_{i} log [exp(sim(z_i, z_i^+)/τ) / ∑_{k} exp(sim(z_i, z_k)/τ)]
                
                其中：
                - z_i, z_i^+：节点 i 在两个不同视图上的表示
                - sim(·, ·)：相似度函数（通常使用余弦相似度或点积）
                - τ：温度参数，控制分布的平滑度
                - 分子：正样本对（同一节点在不同视图上）的相似度
                - 分母：所有样本对的相似度之和
            
            损失函数参数说明：
                - λ = 0.1：lambda 参数，控制对比学习损失的权重比例
                - p = 0.01：伪标签阈值，用于生成伪标签
                - τ = self.temp：温度参数，从配置中读取
            
            步骤说明：
                1. 对用户和物品索引进行去重和类型转换
                2. 在两个增强视图上分别计算节点表示
                3. 拼接用户和物品的表示，形成统一的节点集合
                4. 计算对比学习损失
        """
        # 对用户和物品索引进行去重，并转换为 long 类型
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(self.device)
        
        # 在第一个增强视图上计算用户和物品的嵌入
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        
        # 在第二个增强视图上计算用户和物品的嵌入
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        
        # 拼接两个视图的用户和物品嵌入，形成统一的节点表示集合
        # view1: [用户嵌入(视图1), 物品嵌入(视图1)]
        # view2: [用户嵌入(视图2), 物品嵌入(视图2)]
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        
        # 计算对比学习损失
        # 参数说明：
        #   view1, view2: 两个不同视图上的节点表示
        #   lambda = 0.1: 损失权重参数
        #   p = 0.01: 伪标签阈值
        #   temp = self.temp: 温度参数（τ）
        return scl_loss(view1, view2, 0.1, 0.01, self.temp)