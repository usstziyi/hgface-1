"""
ViT 学习 2：使用 HuggingFace ViT 做图像分类
难度：⭐⭐ 中等

使用 HuggingFace Transformers 的 ViT 模型进行图像分类。
涵盖：模型加载、推理、特征提取、可视化注意力图。

HuggingFace 模型库: https://huggingface.co/models?library=transformers&search=vit
"""

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 一、基础图像分类
# ============================================================

def demo_basic_classification():
    print("=" * 60)
    print("一、基础图像分类")
    print("=" * 60)

    # 加载预训练模型和处理器
    # google/vit-base-patch16-224 模型参数说明:
    # - google: 模型发布组织 (Google Research)
    # - vit: Vision Transformer 架构
    # - base: 模型大小 (base=基础版, 约86M参数; large=大型, 约304M参数)
    # - patch16: 每个patch的大小为16x16像素 (还有patch32等变体)
    # - 224: 输入图像分辨率 224x224 像素
    model_name = "google/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    print(f"模型: {model_name}")
    print(f"类别数: {model.config.num_labels}")
    print(f"图像尺寸: {model.config.image_size}")
    print(f"Patch 尺寸: {model.config.patch_size}")

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print(f"\n输入图像尺寸: {image.size}")

    # 预处理
    # inputs 包含以下信息：
    # - pixel_values: 预处理后的图像张量，shape 为 [batch_size, 3, 224, 224]
    #   经过归一化（ImageNet 均值和标准差）和 resize 到 224x224
    inputs = processor(images=image, return_tensors="pt")
    print(f"预处理后:")
    for key, value in inputs.items():
        print(f"{key}: {value.shape}")
    # [1, 3, 224, 224]

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\n输出键: {outputs.keys()}")
    # 获取预测结果
    logits = outputs.logits
    print(f"Logits shape: {logits.shape}")  # [1, 1000]



    # 获取 Top-5 预测
    probs = torch.softmax(logits, dim=-1) # [1, 1000]
    top5_probs, top5_indices = torch.topk(probs, 5, dim=-1) # [1, 5]
    print(f"Top-5 概率 shape: {top5_probs.shape}")
    print(f"Top-5 索引 shape: {top5_indices.shape}")

    print("\nTop-5 预测:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        label = model.config.id2label[idx.item()]
        print(f"  #{i+1} {idx.item()}-{label}: {prob.item():.4f}")


# ============================================================
# 二、特征提取（用于下游任务）
# ============================================================

def demo_feature_extraction():
    print("\n" + "=" * 60)
    print("二、特征提取（用于下游任务）")
    print("=" * 60)

    model_name = "google/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # 方法 1：使用 output_hidden_states 获取所有层的输出
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # 默认只有 last_hidden_state
    """
        ViT-Base (google/vit-base-patch16-224)
        │
        ├── Embedding 层      → hidden_states[0]
        │   (Patch Embedding + CLS Token + Position Embedding)
        │
        ├── Transformer Block 1   → hidden_states[1]
        ├── Transformer Block 2   → hidden_states[2]
        ├── Transformer Block 3   → hidden_states[3]
        ├── ...
        ├── Transformer Block 11  → hidden_states[11]
        └── Transformer Block 12  → hidden_states[12]

        总计：1 + 12 = 13 层
    """
    hidden_states = outputs.hidden_states
    print(f"隐藏层数量: {len(hidden_states)}")  # 13 (1 embedding + 12 transformer)
    print(f"每层 shape: {hidden_states[0].shape}")  # [1, 197, 768]
    # 197 = 1 (CLS) + 196 (patches)

    # 获取最后一层的 CLS token 作为图像特征
    last_hidden = hidden_states[-1]
    cls_feature = last_hidden[:, 0, :]  # [1, 768]
    print(f"\nCLS 特征 shape: {cls_feature.shape}")

    # 方法 2：使用 vit 模型直接获取
    from transformers import ViTModel

    """
        预训练模型 (google/vit-base-patch16-224)
        ├── vit (ViT 主体)
        ├── classifier (分类头)  ← 有这些权重
        └── pooler (池化层)      ← 没有这个

        ViTModel 期望的结构
        ├── vit (ViT 主体)
        └── pooler              ← 需要这个，但预训练模型没有

        Key                 | Status     |
        --------------------+------------+-
        classifier.weight   | UNEXPECTED |
        classifier.bias     | UNEXPECTED |
        pooler.dense.bias   | MISSING    |
        pooler.dense.weight | MISSING    |
    """
    
    vit_model = ViTModel.from_pretrained(model_name)
    
    with torch.no_grad():
        outputs = vit_model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    
    print(f"\nViTModel 输出:")
    print(f"  last_hidden_state shape: {last_hidden_state.shape}")  # [1, 197, 768]
    print(f"  pooler_output shape: {pooler_output.shape}")  # [1, 768]
    # pooler_output 是 CLS token 经过 LayerNorm + Linear 的结果

    # 方法 3：获取所有 patch 的特征（用于目标检测、分割等）
    patch_features = last_hidden_state[:, 1:, :]  # 去掉 CLS token
    print(f"\nPatch 特征 shape: {patch_features.shape}")  # [1, 196, 768]
    # 可以 reshape 回空间结构
    patch_features_2d = patch_features.view(1, 14, 14, 768)
    print(f"Patch 特征 (2D): {patch_features_2d.shape}")  # [1, 14, 14, 768]


# ============================================================
# 三、可视化注意力图
# ============================================================

def demo_attention_visualization():
    print("\n" + "=" * 60)
    print("三、注意力图分析")
    print("=" * 60)

    model_name = "google/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, 
        attn_implementation="eager"  # 使用 eager 实现，支持输出注意力权重
    )

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # 获取注意力权重
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    print(f"注意力层数: {len(attentions)}")  # 12
    print(f"每层注意力 shape: {attentions[0].shape}")
    # [1, 12, 197, 197]
    # 1 = batch, 12 = heads, 197 = seq_len, 197 = seq_len
    """
        attentions[0].shape  # [1, 12, 197, 197]
                                ↑   ↑   ↑    ↑
                                │   │   │    └─ Key 的序列长度
                                │   │   └─ Query 的序列长度
                                │   └─ 注意力头数 (12 heads)
                                └─ 批次大小
    """

    # 分析 CLS token 的注意力（CLS 关注哪些 patch）
    cls_attention = attentions[-1][0]  # 最后一层的注意力 [12, 197, 197]
    cls_to_patches = cls_attention[:, 0, 1:]  # 最后一层 CLS token 对所有 patch 的注意力 [12, 196]
    
    print(f"\nCLS token 注意力分析:")
    print(f"  注意力头数: {cls_to_patches.shape[0]}")  # 12
    print(f"  关注的 patch 数: {cls_to_patches.shape[1]}")  # 196

    # 平均所有头的注意力:最后一层cls token对所有patch的注意力
    avg_attention = cls_to_patches.mean(dim=0)  # [196]
    print(f"  平均注意力 shape: {avg_attention.shape}")
    
    # 找到注意力最高的 patch
    top_patches = torch.topk(avg_attention, 10)
    print(f"\n  注意力最高的 10 个 patch 索引: {top_patches.indices.tolist()}")
    print(f"  对应的注意力值: {[f'{v:.4f}' for v in top_patches.values.tolist()]}")

    # 将 patch 索引转换为空间位置
    print("\n  高注意力 patch 的空间位置 (row, col):")
    for idx in top_patches.indices[:5]:
        row = idx.item() // 14
        col = idx.item() % 14
        print(f"    Patch {idx.item()}: ({row}, {col})")

    print("\n可视化建议:")
    print("  - 将 196 个注意力值 reshape 为 14x14 的 heatmap")
    print("  - 叠加到原图上可以看到模型关注的区域")
    print("  - 类似 Grad-CAM 的效果")

    # ==================== 可视化注意力图 ====================
    print("\n正在生成注意力可视化...")

    # 平均所有头的注意力:最后一层cls token对所有patch的注意力
    # 1. 将注意力 reshape 为 14x14 的 heatmap
    attention_map = avg_attention.reshape(14, 14).numpy()

    # 2. 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(image)
    axes[0].set_title('original image')
    axes[0].axis('off')

    # 注意力 heatmap
    im = axes[1].imshow(attention_map, cmap='viridis')
    axes[1].set_title('CLS token attention heatmap (14x14)')
    axes[1].set_xlabel('Patch col')
    axes[1].set_ylabel('Patch row')
    plt.colorbar(im, ax=axes[1])

    # 注意力叠加到原图
    axes[2].imshow(image)  # 原图
    # 将 heatmap resize 到原图大小
    attention_resized = np.array(Image.fromarray(attention_map).resize(
        image.size, Image.BILINEAR
    ))
    # 归一化到 0-1
    attention_resized = (attention_resized - attention_resized.min()) / \
                        (attention_resized.max() - attention_resized.min())
    # 叠加 heatmap
    axes[2].imshow(attention_resized, cmap='jet', alpha=0.5) # 叠加注意力 heatmap
    axes[2].set_title('CLS token attention overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
    print("图片已保存: attention_visualization.png")

    # 3. 可视化不同注意力头
    fig2, axes2 = plt.subplots(3, 4, figsize=(16, 12))
    axes2 = axes2.flatten()

    for head_idx in range(12):
        head_attention = cls_to_patches[head_idx].reshape(14, 14).numpy()
        axes2[head_idx].imshow(head_attention, cmap='viridis')
        axes2[head_idx].set_title(f'Head {head_idx + 1}')
        axes2[head_idx].axis('off')

    plt.suptitle('CLS token attention distribution for each head', fontsize=14)
    plt.tight_layout()
    plt.savefig('attention_heads.png', dpi=150, bbox_inches='tight')
    print("图片已保存: attention_heads.png")

    


# ============================================================
# 四、批量推理
# ============================================================

def demo_batch_inference():
    print("\n" + "=" * 60)
    print("四、批量推理")
    print("=" * 60)

    model_name = "google/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # 多张图像
    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫
        "http://images.cocodataset.org/val2017/000000084327.jpg",  # 街道
    ]
    
    images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in urls]
    print(f"图像数量: {len(images)}")

    # 批量预处理
    inputs = processor(images=images, return_tensors="pt")
    print(f"批量输入 shape: {inputs.pixel_values.shape}")  # [2, 3, 224, 224]

    # 批量推理
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    
    # 每张图像的 Top-1 预测
    for i, (url, prob) in enumerate(zip(urls, probs)):
        top_idx = torch.argmax(prob).item()
        top_prob = prob[top_idx].item()
        label = model.config.id2label[top_idx]
        print(f"\n图像 {i+1}: {url}")
        print(f"  预测: {label} ({top_prob:.4f})")


# ============================================================
# 五、不同 ViT 模型对比
# ============================================================

def demo_model_comparison():
    print("\n" + "=" * 60)
    print("五、不同 ViT 模型对比")
    print("=" * 60)

    models_info = [
        ("google/vit-base-patch16-224", "ViT-Base, ImageNet-21k"),
        ("google/vit-large-patch16-224", "ViT-Large, ImageNet-21k"),
        ("google/vit-base-patch32-384", "ViT-Base, patch32, 384x384"),
    ]

    for model_name, desc in models_info:
        try:
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)
            
            num_params = sum(p.numel() for p in model.parameters())
            
            print(f"\n{model_name}")
            print(f"  描述: {desc}")
            print(f"  参数量: {num_params / 1e6:.1f}M")
            print(f"  类别数: {model.config.num_labels}")
            print(f"  图像尺寸: {model.config.image_size}")
            print(f"  Patch 尺寸: {model.config.patch_size}")
            print(f"  隐藏层维度: {model.config.hidden_size}")
            print(f"  层数: {model.config.num_hidden_layers}")
            print(f"  注意力头数: {model.config.num_attention_heads}")
            
        except Exception as e:
            print(f"\n{model_name}: 加载失败 ({e})")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":


    # import inspect
    # from transformers import ViTForImageClassification
    # print(inspect.signature(ViTForImageClassification.forward))

    # demo_basic_classification()
    # demo_feature_extraction()
    demo_attention_visualization()
    # demo_batch_inference()
    # demo_model_comparison()
