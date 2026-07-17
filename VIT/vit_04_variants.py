"""
ViT 学习 4：ViT 变体 — DeiT, Swin Transformer, DINOv2
难度：⭐⭐⭐⭐ 进阶

ViT 的改进版本：
1. DeiT: 数据高效的 ViT 训练
2. Swin Transformer: 层次化 ViT，支持多尺度
3. DINOv2: 自监督学习的 ViT

这些变体解决了原始 ViT 的一些问题：
- 需要大量数据
- 计算复杂度高（O(n²)）
- 缺乏多尺度特征
"""

import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    SwinForImageClassification,
    SwinModel,
)
from PIL import Image
import requests

# ============================================================
# 一、DeiT: Data-efficient Image Transformer
# ============================================================

def demo_deit():
    print("=" * 60)
    print("一、DeiT (Data-efficient Image Transformer)")
    print("=" * 60)

    """
    DeiT 的核心改进：
    1. 蒸馏训练：使用教师模型（如 ResNet）指导训练
    2. 数据增强：更强的数据增强策略
    3. 可以在较小数据集上达到好的效果
    
    论文: "Training data-efficient image transformers & distillation through attention"
    """

    # DeiT 模型在 HuggingFace 上以 ViT 格式提供
    model_name = "facebook/deit-base-distilled-patch16-224"
    
    try:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        
        print(f"模型: {model_name}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"类别数: {model.config.num_labels}")
        
        # 测试推理
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top_idx = torch.argmax(probs).item()
        
        print(f"\n预测: {model.config.id2label[top_idx]} ({probs[0, top_idx]:.4f})")
        
    except Exception as e:
        print(f"DeiT 模型加载失败: {e}")
        print("可以使用 google/vit-base-patch16-224 作为替代")

    print("\nDeiT vs ViT:")
    print("  - DeiT 使用蒸馏训练，数据效率更高")
    print("  - 在 ImageNet-1k 上就能达到好效果（ViT 需要 ImageNet-21k）")
    print("  - 架构与 ViT 基本相同")


# ============================================================
# 二、Swin Transformer: 层次化 ViT
# ============================================================

def demo_swin_transformer():
    print("\n" + "=" * 60)
    print("二、Swin Transformer (层次化 ViT)")
    print("=" * 60)

    """
    Swin Transformer 的核心改进：
    1. 移动窗口注意力：降低计算复杂度 O(n) vs O(n²)
    2. 层次化特征：类似 CNN 的多尺度特征金字塔
    3. 支持密集预测任务（目标检测、分割）
    
    论文: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    """

    # 加载 Swin 模型
    model_name = "microsoft/swin-base-patch4-window7-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = SwinForImageClassification.from_pretrained(model_name)
    
    print(f"模型: {model_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"类别数: {model.config.num_labels}")
    
    # 查看配置
    print(f"\nSwin 配置:")
    print(f"  图像尺寸: {model.config.image_size}")
    print(f"  Patch 尺寸: {model.config.patch_size}")
    print(f"  窗口尺寸: {model.config.window_size}")
    print(f"  嵌入维度: {model.config.embed_dim}")
    print(f"  深度: {model.config.depths}")  # 每层的 block 数
    print(f"  头数: {model.config.num_heads}")  # 每层的注意力头数

    # 测试推理
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
    
    print("\nTop-5 预测:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        label = model.config.id2label[idx.item()]
        print(f"  #{i+1} {label}: {prob.item():.4f}")

    # 获取层次化特征
    print("\n--- Swin 的层次化特征 ---")
    swin_model = SwinModel.from_pretrained(model_name)
    
    with torch.no_grad():
        outputs = swin_model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    print(f"隐藏层数量: {len(hidden_states)}")
    
    for i, hs in enumerate(hidden_states):
        print(f"  Layer {i}: {hs.shape}")
    
    print("\nSwin vs ViT:")
    print("  - Swin 有层次化结构（类似 CNN 的 stage）")
    print("  - 计算复杂度：O(n) vs ViT 的 O(n²)")
    print("  - 支持多尺度特征，适合密集预测任务")
    print("  - 在 COCO 检测/分割上表现更好")


# ============================================================
# 三、Swin 的不同尺寸
# ============================================================

def demo_swin_variants():
    print("\n" + "=" * 60)
    print("三、Swin Transformer 不同尺寸")
    print("=" * 60)

    swin_models = [
        ("microsoft/swin-tiny-patch4-window7-224", "Tiny"),
        ("microsoft/swin-small-patch4-window7-224", "Small"),
        ("microsoft/swin-base-patch4-window7-224", "Base"),
        ("microsoft/swin-large-patch4-window7-224", "Large"),
    ]

    print(f"{'模型':15} {'参数量':10} {'嵌入维度':10} {'层配置':20}")
    print("-" * 60)

    for model_name, size in swin_models:
        try:
            model = SwinForImageClassification.from_pretrained(model_name)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            embed_dim = model.config.embed_dim
            depths = model.config.depths
            
            print(f"{size:15} {params:8.1f}M {embed_dim:10} {str(depths):20}")
            
        except Exception as e:
            print(f"{size:15} 加载失败: {e}")


# ============================================================
# 四、DINOv2: 自监督 ViT
# ============================================================

def demo_dinov2():
    print("\n" + "=" * 60)
    print("四、DINOv2 (自监督 ViT)")
    print("=" * 60)

    """
    DINOv2 的核心特点：
    1. 自监督学习：不需要标注数据
    2. 强大的视觉特征：可用于各种下游任务
    3. 无需微调即可使用（类似 CLIP）
    
    论文: "DINOv2: Learning Robust Visual Features without Supervision"
    GitHub: https://github.com/facebookresearch/dinov2
    """

    # 注意：DINOv2 在 transformers 中可能需要特定版本
    # 这里演示概念，实际使用可能需要安装 facebookresearch/dinov2
    
    print("DINOv2 特点:")
    print("  - 自监督预训练（使用 DINO 方法）")
    print("  - 无需标注数据，从大量无标注图像学习")
    print("  - 特征质量接近 CLIP，但不需要文本")
    print("  - 适合：特征提取、KNN 分类、作为 backbone")
    
    print("\nDINOv2 模型:")
    print("  - dinov2_vits14: Small, 22M 参数")
    print("  - dinov2_vitb14: Base, 86M 参数")
    print("  - dinov2_vitl14: Large, 300M 参数")
    print("  - dinov2_vitg14: Giant, 1.1B 参数")
    
    # 使用示例（概念代码）
    print("\n使用示例（需要安装 facebookresearch/dinov2）:")
    print("""
    import torch
    import dinov2
    
    # 加载模型
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    # 提取特征
    img = ...  # [B, 3, 518, 518]
    features = model(img)  # [B, 768]
    
    # KNN 分类
    # 直接用特征做 KNN，无需微调
    """)

    print("\nDINOv2 vs CLIP:")
    print("  - DINOv2: 纯视觉，自监督")
    print("  - CLIP: 视觉+语言，对比学习")
    print("  - DINOv2 在纯视觉任务上可能更好")
    print("  - CLIP 支持零样本分类（需要文本）")


# ============================================================
# 五、模型对比总结
# ============================================================

def demo_comparison():
    print("\n" + "=" * 60)
    print("五、ViT 变体对比总结")
    print("=" * 60)

    comparison = """
    ┌─────────────────┬──────────────┬──────────────┬──────────────────┐
    │ 模型             │ 核心改进      │ 计算复杂度    │ 适用场景          │
    ├─────────────────┼──────────────┼──────────────┼──────────────────┤
    │ ViT             │ 基础架构      │ O(n²)        │ 图像分类          │
    │                 │ Patch+Trans  │              │ 需要大量数据       │
    ├─────────────────┼──────────────┼──────────────┼──────────────────┤
    │ DeiT            │ 蒸馏训练      │ O(n²)        │ 数据有限场景      │
    │                 │ 数据高效      │              │ ImageNet-1k 可用  │
    ├─────────────────┼──────────────┼──────────────┼──────────────────┤
    │ Swin            │ 移动窗口      │ O(n)         │ 检测/分割         │
    │ Transformer     │ 层次化结构    │              │ 多尺度特征        │
    ├─────────────────┼──────────────┼──────────────┼──────────────────┤
    │ DINOv2          │ 自监督学习    │ O(n²)        │ 特征提取          │
    │                 │ 无监督预训练  │              │ 无需标注数据       │
    ├─────────────────┼──────────────┼──────────────┼──────────────────┤
    │ CLIP ViT        │ 图文对比      │ O(n²)        │ 零样本分类        │
    │                 │ 多模态对齐    │              │ 图文检索          │
    └─────────────────┴──────────────┴──────────────┴──────────────────┘
    
    选择建议：
    - 图像分类（数据充足）: ViT-B/16 或 Swin-Base
    - 图像分类（数据有限）: DeiT 或 Swin-Small
    - 目标检测/分割: Swin-Transformer（必须！）
    - 特征提取（无标注）: DINOv2
    - 零样本分类: CLIP ViT
    """
    print(comparison)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_deit()
    demo_swin_transformer()
    demo_swin_variants()
    demo_dinov2()
    demo_comparison()
