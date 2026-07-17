"""
项目 3：多模态探索 — 使用 CLIP 模型进行零样本图像分类与图文检索
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# 1. 加载模型与处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. 零样本图像分类
# 加载一张测试图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 候选类别
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# 获取相似度分数
logits_per_image = outputs.logits_per_image  # [1, 3]
probs = logits_per_image.softmax(dim=1)
print("=== 零样本图像分类 ===")
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.4f}")

# 3. 图文检索（扩展）
# 多张图像和描述，通过 logits_per_image / logits_per_text 矩阵实现双向检索
print("\n=== 图文检索 ===")
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000084327.jpg",
]
images = [Image.open(requests.get(u, stream=True).raw) for u in urls]
descriptions = ["a photo of cats", "a photo of a city street"]

retrieval_inputs = processor(
    text=descriptions, images=images, return_tensors="pt", padding=True
)
with torch.no_grad():
    retrieval_outputs = model(**retrieval_inputs)

# 图像 -> 文本检索
logits_per_image = retrieval_outputs.logits_per_image
probs_img2txt = logits_per_image.softmax(dim=1)
print("图像 -> 文本检索:")
for i, desc_probs in enumerate(probs_img2txt):
    print(f"  图像 {i}:")
    for j, p in enumerate(desc_probs):
        print(f"    \"{descriptions[j]}\": {p:.4f}")

# 文本 -> 图像检索
logits_per_text = retrieval_outputs.logits_per_text
probs_txt2img = logits_per_text.softmax(dim=1)
print("文本 -> 图像检索:")
for i, img_probs in enumerate(probs_txt2img):
    print(f"  \"{descriptions[i]}\":")
    for j, p in enumerate(img_probs):
        print(f"    图像 {j}: {p:.4f}")
