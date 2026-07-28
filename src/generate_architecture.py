# coding: utf-8
"""
Generate tsRNA-disease architecture diagram for FREEDOM project

This script creates a comprehensive architecture diagram showing:
- Data layer (tsRNA/Disease embeddings, interactions)
- Model layer (KNN graph, GCN propagation, embedding fusion)
- Training layer (BPR loss, regularization)
- Evaluation layer (AUC, AUPR, etc.)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_architecture_diagram():
    """Create the tsRNA-disease architecture diagram"""
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'data': '#E3F2FD',       # 浅蓝色 - 数据层
        'data_edge': '#1976D2',   # 深蓝色边框
        'model': '#E8F5E9',       # 浅绿色 - 模型层
        'model_edge': '#388E3C',  # 深绿色边框
        'train': '#FFF3E0',       # 浅橙色 - 训练层
        'train_edge': '#F57C00',  # 深橙色边框
        'eval': '#F3E5F5',        # 浅紫色 - 评估层
        'eval_edge': '#7B1FA2',   # 深紫色边框
        'arrow': '#424242',       # 箭头颜色
        'text': '#212121',        # 文字颜色
    }
    
    # ==================== 标题 ====================
    ax.text(8, 11.5, 'FREEDOM for tsRNA-Disease Association Prediction', 
            fontsize=18, fontweight='bold', ha='center', va='center', color=colors['text'])
    ax.text(8, 11.0, 'Architecture Overview', 
            fontsize=12, ha='center', va='center', color='gray')
    
    # ==================== 数据层 (底部) ====================
    data_y = 1.5
    box_height = 1.2
    box_width = 3.0
    
    # tsRNA Embeddings
    tsrna_box = FancyBboxPatch((1, data_y), box_width, box_height, 
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=colors['data'], edgecolor=colors['data_edge'], 
                                linewidth=2)
    ax.add_patch(tsrna_box)
    ax.text(2.5, data_y + 0.7, 'tsRNA Embeddings', fontsize=10, fontweight='bold', 
            ha='center', va='center', color=colors['text'])
    ax.text(2.5, data_y + 0.3, '(n_tsRNA × d)', fontsize=8, ha='center', va='center', color='gray')
    
    # Disease Embeddings
    disease_box = FancyBboxPatch((6.5, data_y), box_width, box_height,
                                  boxstyle="round,pad=0.05,rounding_size=0.2",
                                  facecolor=colors['data'], edgecolor=colors['data_edge'],
                                  linewidth=2)
    ax.add_patch(disease_box)
    ax.text(8, data_y + 0.7, 'Disease Embeddings', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(8, data_y + 0.3, '(n_disease × d)', fontsize=8, ha='center', va='center', color='gray')
    
    # Interaction Data
    inter_box = FancyBboxPatch((12, data_y), box_width, box_height,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=colors['data'], edgecolor=colors['data_edge'],
                                linewidth=2)
    ax.add_patch(inter_box)
    ax.text(13.5, data_y + 0.7, 'Interaction Data', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(13.5, data_y + 0.3, '(tsRNA-disease pairs)', fontsize=8, ha='center', va='center', color='gray')
    
    # 数据层标签
    ax.text(0.3, data_y + 0.6, 'Data\nLayer', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=colors['data_edge'], rotation=0)
    
    # ==================== 模型层 (中间) ====================
    model_y = 4.0
    
    # KNN Graph Construction
    knn_box = FancyBboxPatch((1, model_y), box_width, box_height * 1.5,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=colors['model'], edgecolor=colors['model_edge'],
                             linewidth=2)
    ax.add_patch(knn_box)
    ax.text(2.5, model_y + 1.3, 'KNN Graph', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(2.5, model_y + 0.9, 'Construction', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(2.5, model_y + 0.4, 'Feature similarity', fontsize=8, ha='center', va='center', color='gray')
    
    # User-Item Graph (tsRNA-Disease Association Graph)
    ui_box = FancyBboxPatch((6.5, model_y), box_width, box_height * 1.5,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=colors['model'], edgecolor=colors['model_edge'],
                            linewidth=2)
    ax.add_patch(ui_box)
    ax.text(8, model_y + 1.3, 'User-Item Graph', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(8, model_y + 0.9, '(Bipartite)', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(8, model_y + 0.4, 'tsRNA-disease edges', fontsize=8, ha='center', va='center', color='gray')
    
    # GCN Propagation
    gcn_box = FancyBboxPatch((12, model_y), box_width, box_height * 1.5,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=colors['model'], edgecolor=colors['model_edge'],
                             linewidth=2)
    ax.add_patch(gcn_box)
    ax.text(13.5, model_y + 1.3, 'GCN Propagation', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(13.5, model_y + 0.9, f'L layers', fontsize=9, ha='center', va='center', color=colors['text'])
    ax.text(13.5, model_y + 0.4, 'Message passing', fontsize=8, ha='center', va='center', color='gray')
    
    # 模型层标签
    ax.text(0.3, model_y + 0.9, 'Model\nLayer', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['model_edge'])
    
    # ==================== 图卷积传播细节 ====================
    fusion_y = 7.0
    
    # Embedding Fusion Box
    fusion_box = FancyBboxPatch((4.5, fusion_y), 7, box_height,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=colors['model'], edgecolor=colors['model_edge'],
                                 linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(8, fusion_y + 0.7, 'Embedding Aggregation & Fusion', fontsize=11, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(8, fusion_y + 0.3, r'$e_u = \frac{1}{L+1}\sum_{l=0}^{L} e_u^{(l)}$,  $e_i = \frac{1}{L+1}\sum_{l=0}^{L} e_i^{(l)} + h_i^{mm}$', 
            fontsize=9, ha='center', va='center', color='gray')
    
    # ==================== 训练层 ====================
    train_y = 8.5
    
    # BPR Loss
    bpr_box = FancyBboxPatch((2, train_y), 4, box_height,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=colors['train'], edgecolor=colors['train_edge'],
                             linewidth=2)
    ax.add_patch(bpr_box)
    ax.text(4, train_y + 0.7, 'BPR Loss', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(4, train_y + 0.3, r'$-\log\sigma(e_u \cdot e_i^+ - e_u \cdot e_i^-)$', 
            fontsize=8, ha='center', va='center', color='gray')
    
    # Regularization
    reg_box = FancyBboxPatch((10, train_y), 4, box_height,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=colors['train'], edgecolor=colors['train_edge'],
                             linewidth=2)
    ax.add_patch(reg_box)
    ax.text(12, train_y + 0.7, 'Regularization', fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(12, train_y + 0.3, r'$\lambda \|E\|^2$', fontsize=8, ha='center', va='center', color='gray')
    
    # 训练层标签
    ax.text(0.3, train_y + 0.6, 'Training\nLayer', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['train_edge'])
    
    # ==================== 评估层 (顶部) ====================
    eval_y = 10.0
    
    # Metrics Box
    eval_box = FancyBboxPatch((4.5, eval_y), 7, box_height * 0.8,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=colors['eval'], edgecolor=colors['eval_edge'],
                              linewidth=2)
    ax.add_patch(eval_box)
    ax.text(8, eval_y + 0.65, 'Evaluation Metrics', fontsize=11, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(8, eval_y + 0.25, 'AUC  |  AUPR  |  Precision  |  Recall  |  F1', 
            fontsize=9, ha='center', va='center', color='gray')
    
    # 评估层标签
    ax.text(0.3, eval_y + 0.5, 'Eval\nLayer', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['eval_edge'])
    
    # ==================== 数据流箭头 ====================
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=1.5, 
                       connectionstyle='arc3,rad=0')
    
    # 数据层 -> 模型层
    ax.annotate('', xy=(2.5, model_y), xytext=(2.5, data_y + box_height),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(8, model_y), xytext=(8, data_y + box_height),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(13.5, model_y), xytext=(13.5, data_y + box_height),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    # KNN -> User-Item (水平连接)
    ax.annotate('', xy=(6.5, model_y + 0.9), xytext=(4, model_y + 0.9),
                arrowprops=dict(arrowstyle='<->', color=colors['arrow'], lw=1.5))
    
    # User-Item -> GCN
    ax.annotate('', xy=(12, model_y + 0.9), xytext=(9.5, model_y + 0.9),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    # 模型层 -> 融合层
    ax.annotate('', xy=(8, fusion_y), xytext=(8, model_y + box_height * 1.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    # 融合层 -> 训练层
    ax.annotate('', xy=(4, train_y), xytext=(6, fusion_y + box_height),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(12, train_y), xytext=(10, fusion_y + box_height),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    # 训练层 -> 评估层
    ax.annotate('', xy=(8, eval_y), xytext=(8, train_y + box_height),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    # ==================== De Novo 两阶段标注 ====================
    # 添加虚线框表示 De Novo 流程
    denovo_box = FancyBboxPatch((0.8, 0.3), 14.4, 10.2,
                                 boxstyle="round,pad=0.05,rounding_size=0.3",
                                 facecolor='none', edgecolor='#757575',
                                 linewidth=1.5, linestyle='--')
    ax.add_patch(denovo_box)
    ax.text(0.5, 0.5, 'Stage 1: FREEDOM Training', fontsize=8, fontstyle='italic',
            ha='left', va='bottom', color='#757575')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['data_edge'], label='Data Layer'),
        mpatches.Patch(facecolor=colors['model'], edgecolor=colors['model_edge'], label='Model Layer'),
        mpatches.Patch(facecolor=colors['train'], edgecolor=colors['train_edge'], label='Training Layer'),
        mpatches.Patch(facecolor=colors['eval'], edgecolor=colors['eval_edge'], label='Evaluation Layer'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to generate and save the architecture diagram"""
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'images')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成架构图
    print("Generating tsRNA-disease architecture diagram...")
    fig = create_architecture_diagram()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'tsRNA_disease_architecture.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Architecture diagram saved to: {output_path}")
    
    # 同时保存 PDF 版本（矢量图）
    pdf_path = os.path.join(output_dir, 'tsRNA_disease_architecture.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
