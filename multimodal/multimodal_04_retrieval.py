"""
多模态学习 4：构建图文检索系统
难度：⭐⭐⭐ 中高

使用 CLIP 构建一个完整的图文检索系统：
1. 建立图像索引（预计算图像嵌入）
2. 文本搜索图像（Text-to-Image Retrieval）
3. 图像搜索文本（Image-to-Text Retrieval）
4. 评估检索效果（Recall@K）
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
import numpy as np
import time
import os

# ============================================================
# 一、构建图像索引
# ============================================================

class ImageIndex:
    """
    图像索引类：存储图像嵌入，支持快速检索。
    这是搜索引擎/推荐系统的核心组件。
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.embeddings = []   # 图像嵌入列表
        self.metadata = []     # 图像元信息（路径、描述等）

    @torch.no_grad()
    def add_image(self, image, metadata=None):
        """添加单张图像到索引"""
        inputs = self.processor(images=image, return_tensors="pt")
        features = self.model.get_image_features(**inputs).pooler_output  # 获取投影后的512维特征
        print(f"features.shape: {features.shape}")
        features = F.normalize(features, dim=-1)
        self.embeddings.append(features.cpu())  # 保持 [1, 512] 的2D张量
        self.metadata.append(metadata or {})

    @torch.no_grad()
    def add_images_batch(self, images, metadata_list=None):
        """批量添加图像"""
        inputs = self.processor(images=images, return_tensors="pt")
        features = self.model.get_image_features(**inputs).pooler_output
        features = F.normalize(features, dim=-1)
        self.embeddings.append(features.cpu())
        if metadata_list:
            self.metadata.extend(metadata_list)

    def build(self):
        """构建索引（拼接所有嵌入）"""
        self.index = torch.cat(self.embeddings, dim=0)  # [N, D]
        print(f"索引构建完成: {self.index.shape} ({len(self.metadata)} 张图像)")

    @torch.no_grad()
    def search_by_text(self, query, top_k=5):
        """文本搜索图像"""
        inputs = self.processor(text=[query], return_tensors="pt")
        text_features = self.model.get_text_features(**inputs).pooler_output
        print(f"text_features.shape: {text_features.shape}")
        text_features = F.normalize(text_features, dim=-1) # L2 归一化

        # 计算相似度
        similarities = (text_features @ self.index.T).squeeze()  # [N]
        # 限制 top_k 不能超过索引大小
        top_k = min(top_k, len(self.metadata))
        top_indices = similarities.topk(top_k).indices  # 获取相似度最高的 top_k 个结果的索引

        results = []
        # idx 是索引
        for idx in top_indices:
            results.append({
                "index": idx.item(),
                "score": similarities[idx].item(),
                "metadata": self.metadata[idx.item()],
            })
        return results

    @torch.no_grad()
    def search_by_image(self, query_image, top_k=5):
        """图像搜索图像"""
        inputs = self.processor(images=query_image, return_tensors="pt")
        query_features = self.model.get_image_features(**inputs).pooler_output
        query_features = F.normalize(query_features, dim=-1)

        similarities = (query_features @ self.index.T).squeeze()
        # 限制 top_k 不能超过索引大小
        top_k = min(top_k, len(self.metadata))
        top_indices = similarities.topk(top_k).indices

        results = []
        for idx in top_indices:
            results.append({
                "index": idx.item(),
                "score": similarities[idx].item(),
                "metadata": self.metadata[idx.item()],
            })
        return results


def demo_build_index():
    print("=" * 60)
    print("一、构建图像索引")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    index = ImageIndex(model, processor)

    # 模拟一个图像集合（使用 COCO 数据集的多张图像）
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    image_data = [
        {
            "image_path": os.path.join(data_dir, "cat.jpg"),
            "description": "cats on a couch",
            "id": "img_001",
        },
        {
            "image_path": os.path.join(data_dir, "dog.png"),
            "description": "a dog sticking out its tongue",
            "id": "img_002",
        },
    ]

    # 添加图像
    for data in image_data:
        print(f"  添加图像: {data['description']}...")
        image = Image.open(data["image_path"])
        index.add_image(image, metadata=data)

    # 也可以用多张随机图像模拟大规模数据
    print("  添加 18 张模拟图像...")
    for i in range(18):
        image = Image.new("RGB", (224, 224), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        ))
        index.add_image(image, metadata={
            "description": f"synthetic image {i}",
            "id": f"synthetic_{i:03d}",
        })

    index.build()
    return index


# ============================================================
# 二、文本搜索图像
# ============================================================

def demo_text_to_image_search(index):
    print("\n" + "=" * 60)
    print("二、文本搜索图像 (Text → Image)")
    print("=" * 60)

    queries = [
        "a photo of cats sleeping",
        "a busy city street with cars",
        "a beautiful landscape",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        results = index.search_by_text(query, top_k=3)
        for rank, r in enumerate(results):
            print(f"  #{rank+1} [score={r['score']:.4f}] {r['metadata']['description']} (id={r['metadata']['id']})")


# ============================================================
# 三、图像搜索图像
# ============================================================

def demo_image_to_image_search(index):
    print("\n" + "=" * 60)
    print("三、图像搜索图像 (Image → Image)")
    print("=" * 60)

    # 用一张猫的图片搜索相似图像
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    query_image = Image.open(os.path.join(data_dir, "cat.jpg"))

    print(f"查询图像: {os.path.join(data_dir, 'cat.jpg')}")
    results = index.search_by_image(query_image, top_k=5)
    for rank, r in enumerate(results):
        print(f"  #{rank+1} [score={r['score']:.4f}] {r['metadata']['description']}")


# ============================================================
# 四、评估检索效果（Recall@K）
# ============================================================

def demo_evaluation():
    print("\n" + "=" * 60)
    print("四、检索评估 (Recall@K)")
    print("=" * 60)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 构建一个小型评估数据集
    # 每个图像有一个对应的文本描述（ground truth）
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    eval_data = [
        (os.path.join(data_dir, "cat.jpg"), "cats on couch"),
        (os.path.join(data_dir, "dog.png"), "city street scene"),
    ]

    # 构建索引
    index = ImageIndex(model, processor)
    for url, desc in eval_data:
        image = Image.open(url)
        index.add_image(image, metadata={"description": desc, "url": url})

    # 添加干扰图像
    for i in range(18):
        image = Image.new("RGB", (224, 224), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        ))
        index.add_image(image, metadata={"description": f"distractor {i}", "url": f"synthetic_{i}"})

    index.build()

    # 评估 Recall@K
    # Recall@K = 正确结果在前 K 个中的比例
    print("\nRecall@K 评估:")
    for k in [1, 3, 5, 10]:
        correct = 0
        for url, gt_desc in eval_data:
            results = index.search_by_text(gt_desc, top_k=k)
            # 检查 ground truth 是否在前 K 个结果中
            for r in results:
                if r["metadata"]["url"] == url:
                    correct += 1
                    break
        recall = correct / len(eval_data)
        print(f"  Recall@{k}: {recall:.2%}")

    print("\n说明:")
    print("  Recall@1: 第 1 个结果就是正确答案")
    print("  Recall@5: 前 5 个结果中包含正确答案")
    print("  实际系统中，K 通常为 10, 50, 100")


# ============================================================
# 五、性能优化：大规模检索
# ============================================================

def demo_scaling_tips():
    print("\n" + "=" * 60)
    print("五、大规模检索优化技巧")
    print("=" * 60)

    print("""
1. 预计算 + 缓存嵌入
   - 离线计算所有图像嵌入，存为 .npy 或 .faiss 索引
   - 在线只需编码查询文本，做向量检索

2. FAISS 向量索引
   - pip install faiss-cpu
   - 支持百万级向量的毫秒级检索
   - 支持 IVF（倒排索引）、PQ（乘积量化）等加速

3. 批量编码
   - 用 batch 方式编码图像，而非逐张处理

4. 模型选择
   - clip-vit-base-patch32: 快速，512 维
   - clip-vit-large-patch14: 更准，768 维
   - SigLIP: Google 的改进版本

5. 实际系统架构
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  图像库   │ ──→ │ CLIP编码 │ ──→ │ FAISS索引│
   └──────────┘     └──────────┘     └──────────┘
                                          ↑
   ┌──────────┐     ┌──────────┐          │
   │ 查询文本 │ ──→ │ CLIP编码 │ ──→ 向量检索
   └──────────┘     └──────────┘
""")

    # 演示 FAISS 用法（如果安装了的话）
    try:
        import faiss

        print("--- FAISS 示例 ---")
        dim = 512
        num_vectors = 1000

        # 创建索引
        faiss_index = faiss.IndexFlatIP(dim)  # 内积（等价于余弦相似度，如果向量已归一化）

        # 添加向量
        vectors = np.random.randn(num_vectors, dim).astype("float32")
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        faiss_index.add(vectors)
        print(f"FAISS 索引: {faiss_index.ntotal} 个向量, 维度 {dim}")

        # 检索
        query = np.random.randn(1, dim).astype("float32")
        query = query / np.linalg.norm(query)
        distances, indices = faiss_index.search(query, k=5)
        print(f"Top-5 检索结果索引: {indices[0]}")
        print(f"Top-5 相似度分数: {distances[0]}")

    except ImportError:
        print("FAISS 未安装。安装: pip install faiss-cpu")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    index = demo_build_index()
    # demo_text_to_image_search(index)
    demo_image_to_image_search(index)
    # demo_evaluation()
    # demo_scaling_tips()
