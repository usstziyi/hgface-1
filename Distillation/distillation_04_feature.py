"""
模型蒸馏学习 4：特征蒸馏 — 中间层表示对齐
难度：⭐⭐⭐⭐ 进阶

特征蒸馏 (Feature-based Distillation / Hint-based Distillation)：
学生模型拟合教师模型的中间层特征表示。

论文: "FitNets: Hints for Training Deep Neural Networks" (Romero et al., 2015)

与 Logit 蒸馏的区别：
- Logit 蒸馏：只拟合最终输出
- 特征蒸馏：拟合中间层的特征表示，传递更多知识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 一、特征蒸馏的基本思想
# ============================================================

def demo_feature_distillation_concept():
    print("=" * 60)
    print("一、特征蒸馏的基本思想")
    print("=" * 60)

    concept = """
    特征蒸馏的核心思想：
    
    教师模型（大）：
    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
    │ L1  │ →  │ L2  │ →  │ L3  │ →  │ L4  │ → 输出
    │     │    │     │    │     │    │     │
    └─────┘    └─────┘    └─────┘    └─────┘
       ↓          ↓          ↓          ↓
    特征1       特征2       特征3       特征4
    
    学生模型（小）：
    ┌─────┐    ┌─────┐    ┌─────┐
    │ l1  │ →  │ l2  │ →  │ l3  │ → 输出
    │     │    │     │    │     │
    └─────┘    └─────┘    └─────┘
       ↓          ↓          ↓
    拟合特征1   拟合特征2   拟合特征3
    
    关键问题：维度不匹配！
    - 教师特征维度: [B, 512, H, W]
    - 学生特征维度: [B, 128, H, W]
    
    解决方案：
    1. 线性投影 (FitNets)
    2. 1x1 卷积 (FitNets for CNN)
    3. 注意力机制 (Attention Transfer)
    4. 互信息最大化 (CRD)
    """
    print(concept)


# ============================================================
# 二、FitNets：线性投影对齐
# ============================================================

class FitNetsLoss(nn.Module):
    """FitNets 特征蒸馏损失"""
    
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        # 线性投影将学生特征映射到教师维度
        self.regressor = nn.Linear(student_dim, teacher_dim)
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [B, student_dim]
            teacher_features: [B, teacher_dim]
        """
        # 投影学生特征
        projected = self.regressor(student_features)
        
        # L2 损失
        loss = F.mse_loss(projected, teacher_features.detach())
        
        return loss


def demo_fitnets():
    print("\n" + "=" * 60)
    print("二、FitNets：线性投影对齐")
    print("=" * 60)

    # 模拟特征
    batch_size = 8
    teacher_dim = 512
    student_dim = 128
    
    teacher_features = torch.randn(batch_size, teacher_dim)
    student_features = torch.randn(batch_size, student_dim)
    
    print(f"教师特征 shape: {teacher_features.shape}")
    print(f"学生特征 shape: {student_features.shape}")
    
    # 创建损失
    criterion = FitNetsLoss(teacher_dim, student_dim)
    loss = criterion(student_features, teacher_features)
    
    print(f"\nFitNets 损失: {loss.item():.4f}")
    
    print("\nFitNets 特点:")
    print("  - 使用线性投影对齐维度")
    print("  - L2 损失拟合特征")
    print("  - 简单有效")


# ============================================================
# 三、Attention Transfer：注意力图对齐
# ============================================================

class AttentionTransferLoss(nn.Module):
    """注意力迁移损失"""
    
    def __init__(self):
        super().__init__()
    
    def attention_map(self, features):
        """计算注意力图"""
        # features: [B, C, H, W]
        # 对通道维度求和，得到空间注意力图
        att = features.pow(2).mean(dim=1)  # [B, H, W]
        att = att.view(att.size(0), -1)    # [B, H*W]
        att = F.normalize(att, p=2, dim=1) # L2 归一化
        return att
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [B, C_s, H, W]
            teacher_features: [B, C_t, H, W]
        """
        # 计算注意力图
        student_att = self.attention_map(student_features)
        teacher_att = self.attention_map(teacher_features)
        
        # 注意力图对齐损失
        loss = F.mse_loss(student_att, teacher_att.detach())
        
        return loss


def demo_attention_transfer():
    print("\n" + "=" * 60)
    print("三、Attention Transfer：注意力图对齐")
    print("=" * 60)

    # 模拟特征图
    batch_size = 4
    teacher_channels = 512
    student_channels = 128
    height, width = 14, 14
    
    teacher_features = torch.randn(batch_size, teacher_channels, height, width)
    student_features = torch.randn(batch_size, student_channels, height, width)
    
    print(f"教师特征 shape: {teacher_features.shape}")
    print(f"学生特征 shape: {student_features.shape}")
    
    # 计算注意力图
    criterion = AttentionTransferLoss()
    
    teacher_att = criterion.attention_map(teacher_features)
    student_att = criterion.attention_map(student_features)
    
    print(f"\n教师注意力图 shape: {teacher_att.shape}")
    print(f"学生注意力图 shape: {student_att.shape}")
    
    # 计算损失
    loss = criterion(student_features, teacher_features)
    print(f"\n注意力迁移损失: {loss.item():.4f}")
    
    print("\nAttention Transfer 特点:")
    print("  - 不直接拟合特征值")
    print("  - 拟合空间注意力模式")
    print("  - 不受维度限制")
    print("  - 更关注'看哪里'而非'看到什么'")


# ============================================================
# 四、多层特征蒸馏
# ============================================================

class MultiLayerDistillationLoss(nn.Module):
    """多层特征蒸馏损失"""
    
    def __init__(self, teacher_dims, student_dims, layer_weights=None):
        super().__init__()
        
        # 为每层创建投影器
        self.regressors = nn.ModuleList([
            nn.Linear(s_dim, t_dim)
            for t_dim, s_dim in zip(teacher_dims, student_dims)
        ])
        
        # 层权重
        if layer_weights is None:
            layer_weights = [1.0] * len(teacher_dims)
        self.layer_weights = layer_weights
    
    def forward(self, student_features_list, teacher_features_list):
        """
        Args:
            student_features_list: 学生各层特征列表
            teacher_features_list: 教师各层特征列表
        """
        total_loss = 0
        
        for i, (s_feat, t_feat) in enumerate(
            zip(student_features_list, teacher_features_list)
        ):
            # 投影学生特征
            projected = self.regressors[i](s_feat)
            
            # L2 损失
            layer_loss = F.mse_loss(projected, t_feat.detach())
            
            # 加权
            total_loss += self.layer_weights[i] * layer_loss
        
        return total_loss


def demo_multi_layer_distillation():
    print("\n" + "=" * 60)
    print("四、多层特征蒸馏")
    print("=" * 60)

    # 模拟多层特征
    batch_size = 8
    
    # 教师 4 层特征
    teacher_dims = [256, 512, 1024, 2048]
    teacher_features = [
        torch.randn(batch_size, dim) for dim in teacher_dims
    ]
    
    # 学生 4 层特征
    student_dims = [64, 128, 256, 512]
    student_features = [
        torch.randn(batch_size, dim) for dim in student_dims
    ]
    
    print("教师特征维度:")
    for i, feat in enumerate(teacher_features):
        print(f"  Layer {i+1}: {feat.shape}")
    
    print("\n学生特征维度:")
    for i, feat in enumerate(student_features):
        print(f"  Layer {i+1}: {feat.shape}")
    
    # 创建损失
    criterion = MultiLayerDistillationLoss(
        teacher_dims=teacher_dims,
        student_dims=student_dims,
        layer_weights=[0.5, 0.75, 1.0, 1.0]  # 更重视深层
    )
    
    loss = criterion(student_features, teacher_features)
    print(f"\n多层蒸馏损失: {loss.item():.4f}")
    
    print("\n多层蒸馏特点:")
    print("  - 同时拟合多个中间层")
    print("  - 可以为不同层设置不同权重")
    print("  - 传递更丰富的知识")


# ============================================================
# 五、特征蒸馏 vs Logit 蒸馏
# ============================================================

def demo_comparison():
    print("\n" + "=" * 60)
    print("五、特征蒸馏 vs Logit 蒸馏")
    print("=" * 60)

    comparison = """
    蒸馏方法对比：
    
    ┌─────────────────┬──────────────────┬──────────────────┐
    │ 特性             │ Logit 蒸馏       │ 特征蒸馏          │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ 拟合目标         │ 最终输出         │ 中间层特征        │
    │ 信息量           │ 较少（只有输出） │ 丰富（多层特征）  │
    │ 实现复杂度       │ 简单             │ 较复杂            │
    │ 维度对齐         │ 不需要           │ 需要投影器        │
    │ 计算开销         │ 低               │ 中等              │
    │ 适用场景         │ 通用             │ 需要精细控制      │
    │ 典型方法         │ Hinton 2015      │ FitNets, AT       │
    └─────────────────┴──────────────────┴──────────────────┘
    
    组合使用：
    - 很多工作同时使用 Logit 蒸馏 + 特征蒸馏
    - 两者互补，效果更好
    - 例如：PKT (Probabilistic Knowledge Transfer)
    """
    print(comparison)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_feature_distillation_concept()
    demo_fitnets()
    demo_attention_transfer()
    demo_multi_layer_distillation()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    特征蒸馏的核心要点：
    
    1. 拟合中间层特征，传递更多知识
    
    2. 主要方法：
       - FitNets: 线性投影 + L2 损失
       - Attention Transfer: 注意力图对齐
       - 多层蒸馏: 同时拟合多层
    
    3. 关键挑战：维度不匹配
       - 需要投影器对齐维度
       - 或使用与维度无关的方法（如注意力图）
    
    4. 与 Logit 蒸馏互补
       - 可以组合使用
       - 获得更好的蒸馏效果
    """)
