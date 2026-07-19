"""
模型蒸馏学习 3：视觉蒸馏 — DeiT 风格 ViT 蒸馏
难度：⭐⭐⭐ 中高

DeiT (Data-efficient Image Transformer) 使用蒸馏训练 ViT。
核心创新：引入蒸馏 token，教师指导学生学习。

论文: "Training data-efficient image transformers & distillation through attention"
GitHub: https://github.com/facebookresearch/deit

架构对比：
- 标准 ViT: [CLS] + patches → Transformer → 分类
- DeiT: [CLS] + [DIST] + patches → Transformer → 分类 + 蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# ============================================================
# 一、DeiT 蒸馏 Token 机制
# ============================================================

def demo_distillation_token():
    print("=" * 60)
    print("一、DeiT 蒸馏 Token 机制")
    print("=" * 60)

    print("""
    DeiT 的核心创新：蒸馏 Token
    
    标准 ViT:
    ┌─────────┬──────┬──────┬─────┬──────┐
    │ [CLS]   │ P1   │ P2   │ ... │ P196 │
    └─────────┴──────┴──────┴─────┴──────┘
         ↓
    分类头 → 预测
    
    DeiT:
    ┌─────────┬─────────┬──────┬──────┬─────┬──────┐
    │ [CLS]   │ [DIST]  │ P1   │ P2   │ ... │ P196 │
    └─────────┴─────────┴──────┴──────┴─────┴──────┘
         ↓         ↓
    分类头    蒸馏头 → 拟合教师输出
    
    [DIST] token:
    - 专门用于蒸馏的可学习 token
    - 与 [CLS] token 并行
    - 学习教师的知识
    - 推理时只用 [CLS] token
    """)

    # 模拟 DeiT 的输入序列
    batch_size = 1
    embed_dim = 768
    
    cls_token = torch.randn(1, 1, embed_dim)
    dist_token = torch.randn(1, 1, embed_dim)  # 蒸馏 token
    patch_tokens = torch.randn(1, 196, embed_dim)
    
    # 拼接
    input_tokens = torch.cat([cls_token, dist_token, patch_tokens], dim=1)
    print(f"\n输入序列 shape: {input_tokens.shape}")  # [1, 198, 768]
    print(f"  198 = 1 (CLS) + 1 (DIST) + 196 (patches)")


# ============================================================
# 二、DeiT 风格的蒸馏损失
# ============================================================

class DeiTDistillationLoss(nn.Module):
    """DeiT 风格的蒸馏损失"""
    
    def __init__(self, temperature=3.0, alpha=0.5, distillation_type='soft'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_type = distillation_type
    
    def forward(self, student_cls, student_dist, teacher_logits, labels):
        """
        Args:
            student_cls: 学生的 CLS token 输出
            student_dist: 学生的 DIST token 输出
            teacher_logits: 教师的输出
            labels: 真实标签
        """
        # 硬标签损失（基于 CLS token）
        loss_hard = F.cross_entropy(student_cls, labels)
        
        # 蒸馏损失（基于 DIST token）
        if self.distillation_type == 'soft':
            # 软标签蒸馏
            T = self.temperature
            loss_distill = F.kl_div(
                F.log_softmax(student_dist / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
        else:
            # 硬标签蒸馏（教师预测作为硬标签）
            teacher_pred = teacher_logits.argmax(dim=-1)
            loss_distill = F.cross_entropy(student_dist, teacher_pred)
        
        # 总损失
        loss = (1 - self.alpha) * loss_hard + self.alpha * loss_distill
        
        return loss, loss_hard, loss_distill


def demo_deit_loss():
    print("\n" + "=" * 60)
    print("二、DeiT 风格的蒸馏损失")
    print("=" * 60)

    # 模拟数据
    batch_size = 4
    num_classes = 1000
    
    student_cls = torch.randn(batch_size, num_classes)
    student_dist = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失
    criterion = DeiTDistillationLoss(temperature=3.0, alpha=0.5, distillation_type='soft')
    loss, loss_hard, loss_distill = criterion(
        student_cls, student_dist, teacher_logits, labels
    )
    
    print(f"硬标签损失: {loss_hard.item():.4f}")
    print(f"蒸馏损失: {loss_distill.item():.4f}")
    print(f"总损失: {loss.item():.4f}")
    
    print("\nDeiT 蒸馏特点:")
    print("  - CLS token 负责分类（硬标签损失）")
    print("  - DIST token 负责蒸馏（软标签损失）")
    print("  - 两个目标解耦，互不干扰")
    print("  - 推理时只用 CLS token")


# ============================================================
# 三、使用 HuggingFace ViT 进行蒸馏
# ============================================================

def demo_hf_vit_distillation():
    print("\n" + "=" * 60)
    print("三、使用 HuggingFace ViT 进行蒸馏")
    print("=" * 60)

    # 教师模型
    teacher_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(teacher_name)
    teacher = ViTForImageClassification.from_pretrained(teacher_name)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"教师模型: {teacher_name}")
    print(f"教师参数量: {teacher_params / 1e6:.1f}M")
    
    # 学生模型（更小的 ViT）
    student_name = "google/vit-base-patch32-224"  # patch_size=32，更快
    student = ViTForImageClassification.from_pretrained(student_name)
    
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\n学生模型: {student_name}")
    print(f"学生参数量: {student_params / 1e6:.1f}M")
    print(f"压缩比: {teacher_params / student_params:.2f}x")
    
    # 模拟数据
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # 教师推理
    teacher.eval()
    with torch.no_grad():
        teacher_outputs = teacher(**inputs)
        teacher_logits = teacher_outputs.logits
    
    print(f"\n教师 logits shape: {teacher_logits.shape}")
    
    # 学生训练
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-5)
    
    # 前向传播
    student_outputs = student(**inputs)
    student_logits = student_outputs.logits
    
    # 蒸馏损失
    T = 5.0
    alpha = 0.7
    
    loss_hard = F.cross_entropy(
        student_logits,
        torch.tensor([285])  # 假设真实标签是 285
    )
    loss_soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)
    
    loss = alpha * loss_soft + (1 - alpha) * loss_hard
    
    print(f"\n蒸馏训练:")
    print(f"  硬标签损失: {loss_hard.item():.4f}")
    print(f"  软标签损失: {loss_soft.item():.4f}")
    print(f"  总损失: {loss.item():.4f}")


# ============================================================
# 四、DeiT 的训练策略
# ============================================================

def demo_deit_training_strategy():
    print("\n" + "=" * 60)
    print("四、DeiT 的训练策略")
    print("=" * 60)

    strategy = """
    DeiT 的数据高效训练策略：
    
    1. 强数据增强
       - RandAugment: 随机选择增强操作
       - CutMix: 混合两张图像
       - Mixup: 线性插值两张图像
       - Repeated Augmentation: 多次增强同一图像
    
    2. 蒸馏训练
       - 使用预训练的教师模型（如 ResNet-50）
       - DIST token 学习教师的软标签
       - 结合硬标签和软标签
    
    3. 正则化
       - 权重衰减: 0.3
       - Dropout: 0.0
       - 高学习率配合 cosine schedule
    
    4. 训练效率
       - 在 ImageNet-1K 上就能训练（不需要 ImageNet-21K）
       - 36 小时 × 16 GPU = 576 GPU 小时
       - 达到 83.1% Top-1 准确率
    """
    print(strategy)


# ============================================================
# 五、DeiT 模型变体
# ============================================================

def demo_deit_variants():
    print("\n" + "=" * 60)
    print("五、DeiT 模型变体")
    print("=" * 60)

    variants = """
    DeiT 模型对比：
    
    ┌─────────────────┬──────────┬──────────┬──────────────┐
    │ 模型             │ 参数量    │ Top-1    │ 说明          │
    ├─────────────────┼──────────┼──────────┼──────────────┤
    │ DeiT-Tiny       │ 5.7M     │ 72.2%    │ 最小版本      │
    │ DeiT-Small      │ 22M      │ 79.9%    │ 平衡版本      │
    │ DeiT-Base       │ 86M      │ 83.1%    │ 标准版本      │
    │ DeiT-Base (蒸馏) │ 86M     │ 85.2%    │ + 蒸馏提升    │
    └─────────────────┴──────────┴──────────┴──────────────┘
    
    蒸馏带来的提升：
    - DeiT-Base: 83.1% → 85.2% (+2.1%)
    - 使用相同的参数量，只增加一个 DIST token
    - 蒸馏是免费的性能提升！
    
    HuggingFace 上的 DeiT 模型：
    - facebook/deit-base-distilled-patch16-224
    - facebook/deit-base-patch16-224
    """
    print(variants)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_distillation_token()
    demo_deit_loss()
    demo_hf_vit_distillation()
    demo_deit_training_strategy()
    demo_deit_variants()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    DeiT 蒸馏的核心要点：
    
    1. 蒸馏 Token (DIST)
       - 专门用于蒸馏的可学习 token
       - 与 CLS token 并行，互不干扰
    
    2. 双损失设计
       - CLS token: 硬标签损失（分类）
       - DIST token: 软标签损失（蒸馏）
    
    3. 数据高效
       - 强数据增强
       - 在 ImageNet-1K 上就能训练
    
    4. 免费性能提升
       - 相同参数量，蒸馏带来 2%+ 提升
       - 推理时只用 CLS token，无额外开销
    """)
