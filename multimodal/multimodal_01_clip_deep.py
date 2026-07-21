"""
多模态学习 1：CLIP 图文对齐深入
难度：⭐ 基础

CLIP (Contrastive Language-Image Pre-training) 是多模态的基石模型。
它通过对比学习，将图像和文本映射到同一向量空间。

核心概念：
- 图像编码器：Vision Transformer (ViT)
- 文本编码器：Transformer
- 对比损失：让匹配的图文对靠近，不匹配的远离
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
from PIL import Image
import requests
import numpy as np

# ============================================================
# 一、CLIP 基础：理解图文嵌入
# ============================================================

def demo_clip_embeddings():
    print("=" * 60)
    print("一、CLIP 图文嵌入（Embedding）")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 准备输入
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True
    )

    # 获取嵌入向量
    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image 和 logits_per_text 互为转置
    outputs.logits_per_image  # [1, 2] 图像对两个文本的匹配分数
    outputs.logits_per_text   # [2, 1] 两个文本对图像的匹配分数（转置）

    # 图像嵌入 [1, 512]
    image_embeds = outputs.image_embeds
    print(f"图像嵌入 shape: {image_embeds.shape}")  # [1, 512]
    print(f"图像嵌入 L2 范数: {torch.norm(image_embeds).item():.4f}")

    # 文本嵌入 [2, 512]
    text_embeds = outputs.text_embeds
    print(f"文本嵌入 shape: {text_embeds.shape}")  # [2, 512]

    # outputs.image_embeds 和 outputs.text_embeds 已经是 L2 归一化后的向量。

    # 计算余弦相似度
    similarity = F.cosine_similarity(image_embeds, text_embeds)
    print(f"\n图像与 'a photo of a cat' 相似度: {similarity[0].item():.4f}")
    print(f"图像与 'a photo of a dog' 相似度: {similarity[1].item():.4f}")

    # 1.归一化
    # 2.点积 = 余弦相似度
    # CLIP 的 logits 就是这个相似度乘以温度系数
    print(f"\n模型输出的 logits_per_image: {outputs.logits_per_image}")
    print(f"模型输出的 logits_per_text: {outputs.logits_per_text}")


# ============================================================
# 二、零样本分类：用文本描述作为类别
# ============================================================

def demo_zero_shot_classification():
    print("\n" + "=" * 60)
    print("二、零样本图像分类")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 多张图像
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    image_files = [
        os.path.join(data_dir, "cat.jpg"),
        os.path.join(data_dir, "dog.png"),
    ]
    images = [Image.open(image_file) for image_file in image_files]

    # 类别描述（可以用自然语言！）
    candidate_labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car",
        "a photo of a city street",
        "a photo of food",
    ]

    # 批量处理
    inputs = processor(
        text=candidate_labels,
        images=images,
        return_tensors="pt",
        padding=True
    )

    """inputs
    # 图像编码器内部
    pixel_values          # [batch, 3, 224, 224] 原始像素
        ↓
    patch_embedding       # 卷积层：将图像切成 patch 并嵌入
        ↓
    position_embedding    # 添加位置编码
        ↓
    transformer_layers    # Transformer 处理
        ↓
    image_embeds          # [batch, 512] 最终嵌入

    # 文本编码器内部
    input_ids             # [batch, seq_len] token 索引
        ↓
    token_embedding       # Embedding 查表：ID → 向量
        ↓
    position_embedding    # 添加位置编码
        ↓
    transformer_layers    # Transformer 处理
        ↓
    text_embeds           # [batch, 512] 最终嵌入
    """

    with torch.no_grad():
        outputs = model(**inputs)

    """ outputs
    dict_keys([
        'logits_per_image',    # 图像对文本的匹配分数
        'logits_per_text',     # 文本对图像的匹配分数
        'text_embeds',         # 文本嵌入向量
        'image_embeds',        # 图像嵌入向量
        'text_model_output',   # 文本编码器的完整输出
        'vision_model_output'  # 图像编码器的完整输出
    ])
    """

    # logits_per_image: [num_images, num_labels]
    logits_per_image = outputs.logits_per_image # [2, 5]
    probs = logits_per_image.softmax(dim=1) # [2, 5]

    print(f"logits_per_image shape: {logits_per_image.shape}")  # [2, 5]

    for i, (image_file, prob) in enumerate(zip(image_files, probs)):
        print(f"\n图像 {i+1}: {image_file}")
        # 排序显示
        sorted_indices = prob.argsort(descending=True)
        # rank=排名，idx=原始索引(idx)
        for rank, idx in enumerate(sorted_indices):
            label = candidate_labels[idx]
            score = prob[idx].item()
            print(f"  #{rank+1} {label}: {score:.4f}")


# ============================================================
# 三、图像特征提取：用于下游任务
# ============================================================

def demo_feature_extraction():
    print("\n" + "=" * 60)
    print("三、图像特征提取（用于下游任务）")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 方法 1：通过 processor + model
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    print(f"图像特征 shape: {image_features.shape}")  # [1, 512]

    # 方法 2：只使用视觉编码器
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs.pixel_values
        )
        # pooler_output 是经过投影的图像特征
        pooled = vision_outputs.pooler_output
        print(f"Vision encoder pooled shape: {pooled.shape}")  # [1, 512]

        # last_hidden_state 是所有 patch 的特征
        patch_features = vision_outputs.last_hidden_state
        print(f"Patch features shape: {patch_features.shape}")  # [1, 50, 768]
        # 50 = 1 (CLS) + 49 (7x7 patches)

    # 文本特征提取
    text_inputs = processor(text=["a photo of a cat"], return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    print(f"\n文本特征 shape: {text_features.shape}")  # [1, 512]

    # 特征归一化（CLIP 内部会做）
    image_features_norm = F.normalize(image_features, dim=-1)
    text_features_norm = F.normalize(text_features, dim=-1)
    similarity = (image_features_norm @ text_features_norm.T).item()
    print(f"归一化后的相似度: {similarity:.4f}")


# ============================================================
# 四、构建图像-文本相似度矩阵
# ============================================================

def demo_similarity_matrix():
    print("\n" + "=" * 60)
    print("四、图像-文本相似度矩阵")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 3 张图像
    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000084327.jpg",
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in urls]

    # 3 条文本描述
    texts = [
        "a photo of cats",
        "a photo of a city street",
        "a photo of a delicious meal",
    ]

    # 分别编码图像和文本
    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = processor(text=texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

    # 归一化
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 计算相似度矩阵 [num_images, num_texts]
    similarity_matrix = image_features @ text_features.T
    print(f"相似度矩阵 shape: {similarity_matrix.shape}")  # [2, 3]

    print("\n相似度矩阵:")
    print(f"{'':20}", end="")
    for t in texts:
        print(f"{t[:15]:15}", end="")
    print()

    for i, url in enumerate(urls):
        print(f"{url[:18]:20}", end="")
        for j in range(len(texts)):
            print(f"{similarity_matrix[i, j].item():.4f}       ", end="")
        print()

    # 这就是图文检索的基础！
    print("\n应用：")
    print("  - 图像检索：给定文本，找最相似的图像")
    print("  - 文本检索：给定图像，找最相似的描述")


# ============================================================
# 五、CLIP 的 Prompt Engineering
# ============================================================

def demo_prompt_engineering():
    print("\n" + "=" * 60)
    print("五、CLIP Prompt Engineering")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 不同的 prompt 模板会影响分类效果
    class_names = ["cat", "dog", "car", "bird"]

    # 模板 1：简单描述
    prompts_simple = [f"a photo of a {c}" for c in class_names]

    # 模板 2：更详细的描述
    prompts_detailed = [f"a photo of a {c} in a natural setting" for c in class_names]

    # 模板 3：使用多个模板集成（更鲁棒）
    templates = [
        "a photo of a {}.",
        "a photograph of a {}.",
        "an image of a {}.",
        "a blurry photo of a {}.",
        "a close-up photo of a {}.",
    ]
    prompts_ensemble = []
    for c in class_names:
        for t in templates:
            prompts_ensemble.append(t.format(c))

    print(f"简单 prompt: {prompts_simple}")
    print(f"集成 prompt 数量: {len(prompts_ensemble)}")

    # 测试不同 prompt
    for name, prompts in [("简单", prompts_simple), ("集成", prompts_ensemble)]:
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image

        if name == "集成":
            # 集成：对每个类别的多个 prompt 取平均
            logits = logits.reshape(len(class_names), len(templates)).mean(dim=1)

        probs = logits.softmax(dim=1)
        print(f"\n{name} prompt 结果:")
        for c, p in zip(class_names, probs[0]):
            print(f"  {c}: {p.item():.4f}")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # demo_clip_embeddings()
    demo_zero_shot_classification()
    # demo_feature_extraction()
    # demo_similarity_matrix()
    # demo_prompt_engineering()
