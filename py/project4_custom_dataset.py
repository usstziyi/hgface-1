"""
项目 4：自定义 Dataset 与 DataLoader — 处理文本和图像
学习 torch.utils.data.Dataset 和 DataLoader 的核心用法。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import requests
import io
import os

# ============================================================
# 一、文本 Dataset：情感分析数据
# ============================================================

class SentimentDataset(Dataset):
    """
    自定义文本分类 Dataset。
    接收原始文本和标签列表，使用 tokenizer 进行编码。
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: 文本列表 ["I love this movie", "terrible film", ...]
            labels: 标签列表 [1, 0, ...]
            tokenizer: Hugging Face 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 对单条文本编码
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 返回字典，每个 value 去掉 batch 维度（squeeze）
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def demo_text_dataset():
    print("=" * 60)
    print("一、文本 Dataset 演示")
    print("=" * 60)

    # 模拟数据
    texts = [
        "This movie is fantastic! I really enjoyed it.",
        "Terrible film, a complete waste of time.",
        "Absolutely wonderful, highly recommend!",
        "Boring and predictable, did not like it.",
        "One of the best movies I have ever seen.",
        "Awful acting and a weak storyline.",
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1=正面, 0=负面

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 创建 Dataset
    dataset = SentimentDataset(texts, labels, tokenizer)
    print(f"数据集大小: {len(dataset)}")

    # 取单条数据查看
    sample = dataset[0]
    print(f"\n单条样本的 keys: {list(sample.keys())}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"标签: {sample['label']}")

    # 创建 DataLoader，自动处理 batch 和 shuffle
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"\nDataLoader 批次数: {len(dataloader)}")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"  input_ids shape: {batch['input_ids'].shape}")       # [batch_size, max_length]
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['label']}")

    # 配合模型训练
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# ============================================================
# 二、图像 Dataset：加载并预处理图像
# ============================================================

class ImageTextDataset(Dataset):
    """
    自定义图像 Dataset。
    接收图像路径和标签，加载图像并进行预处理。
    可用于图像分类、图文匹配等任务。
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 图像预处理（如 torchvision.transforms）
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像（如果路径不存在则用随机图像代替）
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # 演示用：生成随机图像
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


def demo_image_dataset():
    print("\n" + "=" * 60)
    print("二、图像 Dataset 演示")
    print("=" * 60)

    # 模拟图像路径（本地不存在时会生成随机图像）
    image_paths = [
        "./images/cat.jpg",
        "./images/dog.jpg",
        "./images/car.jpg",
        "./images/cat2.jpg",
    ]
    labels = [0, 1, 2, 0]  # 0=cat, 1=dog, 2=car

    # 使用 torchvision 风格的 transform（这里用 torchvision 的 Compose）
    try:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    except ImportError:
        # 没有 torchvision 时用简单的手动预处理
        def transform(image):
            image = image.resize((224, 224))
            import numpy as np
            arr = np.array(image, dtype=np.float32) / 255.0
            # HWC -> CHW
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            return tensor

    dataset = ImageTextDataset(image_paths, labels, transform=transform)
    print(f"数据集大小: {len(dataset)}")

    sample = dataset[0]
    print(f"\n单条样本的 keys: {list(sample.keys())}")
    print(f"pixel_values shape: {sample['pixel_values'].shape}")  # [3, 224, 224]
    print(f"标签: {sample['label']}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"\nDataLoader 批次数: {len(dataloader)}")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"  pixel_values shape: {batch['pixel_values'].shape}")  # [2, 3, 224, 224]
        print(f"  labels: {batch['label']}")


# ============================================================
# 三、结合 CLIP Processor 的图文 Dataset
# ============================================================

class CLIPTripletDataset(Dataset):
    """
    图文配对 Dataset，使用 CLIPProcessor 同时处理图像和文本。
    适用于图文对比学习、图文检索等任务。
    """

    def __init__(self, image_urls, captions, processor):
        self.image_urls = image_urls
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        # 下载图像
        url = self.image_urls[idx]
        caption = self.captions[idx]

        try:
            image = Image.open(requests.get(url, stream=True, timeout=10).raw).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # 用 CLIP processor 处理单条图文对
        inputs = self.processor(
            text=caption, images=image, return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
        }


def demo_clip_dataset():
    print("\n" + "=" * 60)
    print("三、CLIP 图文配对 Dataset 演示")
    print("=" * 60)

    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000084327.jpg",
    ]
    captions = [
        "a photo of cats sleeping on a couch",
        "a photo of a city street with cars",
    ]

    dataset = CLIPTripletDataset(image_urls, captions, processor)
    print(f"数据集大小: {len(dataset)}")

    sample = dataset[0]
    print(f"\n单条样本的 keys: {list(sample.keys())}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"pixel_values shape: {sample['pixel_values'].shape}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  pixel_values shape: {batch['pixel_values'].shape}")


# ============================================================
# 运行所有演示
# ============================================================

if __name__ == "__main__":
    demo_text_dataset()
    demo_image_dataset()
    demo_clip_dataset()
