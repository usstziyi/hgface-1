"""
项目 7：PyTorch Dataset 与 HuggingFace Dataset 互转
涵盖两个方向：
  1. HuggingFace Dataset → PyTorch Dataset / DataLoader
  2. PyTorch Dataset / TensorDataset → HuggingFace Dataset
"""

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset, DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ============================================================
# 一、HuggingFace Dataset → PyTorch DataLoader（直接兼容）
# ============================================================

def demo_hf_to_torch_dataloader():
    print("=" * 60)
    print("一、HuggingFace Dataset → PyTorch DataLoader")
    print("=" * 60)

    # HuggingFace Dataset 本身就可以直接传给 PyTorch DataLoader
    dataset = load_dataset("imdb", split="train[:200]")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # 设置 PyTorch 格式：告诉 dataset 返回 torch.Tensor
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print(f"HF Dataset 列: {tokenized.column_names}")
    print(f"HF Dataset 格式: {tokenized.format}")

    # 直接传给 PyTorch DataLoader（完全兼容！）
    dataloader = DataLoader(tokenized, batch_size=16, shuffle=True)

    for batch in dataloader:
        print(f"\nDataLoader batch keys: {list(batch.keys())}")
        print(f"  input_ids: {batch['input_ids'].shape}, dtype: {batch['input_ids'].dtype}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  label: {batch['label'].shape}")
        break

    print("\n关键方法: dataset.set_format(type='torch')")
    print("  让 HF Dataset 返回 torch.Tensor 而非 Python list/dict")


# ============================================================
# 二、HuggingFace Dataset → PyTorch 自定义 Dataset
# ============================================================

class HFDatasetWrapper(TorchDataset):
    """
    将 HuggingFace Dataset 包装为 PyTorch Dataset。
    适用于需要在 __getitem__ 中做额外处理的场景。
    """

    def __init__(self, hf_dataset, tokenizer, max_length=64):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def demo_hf_to_torch_custom():
    print("\n" + "=" * 60)
    print("二、HuggingFace Dataset → PyTorch 自定义 Dataset")
    print("=" * 60)

    hf_dataset = load_dataset("imdb", split="train[:100]")

    # 包装为 PyTorch Dataset
    torch_dataset = HFDatasetWrapper(hf_dataset, tokenizer, max_length=64)
    print(f"PyTorch Dataset 大小: {len(torch_dataset)}")

    sample = torch_dataset[0]
    print(f"单条样本 keys: {list(sample.keys())}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")

    dataloader = DataLoader(torch_dataset, batch_size=8, shuffle=True)
    for batch in dataloader:
        print(f"\nBatch keys: {list(batch.keys())}")
        print(f"  input_ids: {batch['input_ids'].shape}")
        break

    print("\n这种方式的好处: __getitem__ 中可以做任意自定义处理")
    print("  比如数据增强、多模态处理、条件过滤等")


# ============================================================
# 三、HuggingFace Dataset → TensorDataset（纯 tensor）
# ============================================================

def demo_hf_to_tensor_dataset():
    print("\n" + "=" * 60)
    print("三、HuggingFace Dataset → TensorDataset")
    print("=" * 60)

    hf_dataset = load_dataset("imdb", split="train[:100]")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 设置为 PyTorch 格式
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 提取所有 tensor
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = tokenized["label"]

    print(f"input_ids type: {type(input_ids)}, shape: {input_ids.shape}")
    print(f"attention_mask type: {type(attention_mask)}, shape: {attention_mask.shape}")
    print(f"labels type: {type(labels)}, shape: {labels.shape}")

    # 创建 TensorDataset
    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    print(f"\nTensorDataset 大小: {len(tensor_dataset)}")

    sample_x, sample_mask, sample_y = tensor_dataset[0]
    print(f"单条样本: x={sample_x.shape}, mask={sample_mask.shape}, y={sample_y}")

    dataloader = DataLoader(tensor_dataset, batch_size=16, shuffle=True)
    for batch_x, batch_mask, batch_y in dataloader:
        print(f"\nBatch: x={batch_x.shape}, mask={batch_mask.shape}, y={batch_y.shape}")
        break

    print("\n转换路径: HF Dataset → map 分词 → set_format('torch') → 提取 tensor → TensorDataset")


# ============================================================
# 四、PyTorch TensorDataset → HuggingFace Dataset
# ============================================================

def demo_tensor_to_hf():
    print("\n" + "=" * 60)
    print("四、PyTorch TensorDataset → HuggingFace Dataset")
    print("=" * 60)

    # 1. 先创建一个 PyTorch TensorDataset
    input_ids = torch.randint(0, 30000, (100, 64))
    attention_mask = torch.ones(100, 64, dtype=torch.long)
    labels = torch.randint(0, 2, (100,))

    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    print(f"PyTorch TensorDataset 大小: {len(tensor_dataset)}")

    # 2. 转换为 HuggingFace Dataset
    # 方法：将 tensor 转为 dict，再用 Dataset.from_dict()
    hf_dataset = Dataset.from_dict({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "label": labels.tolist(),
    })
    print(f"\nHF Dataset: {hf_dataset}")
    print(f"  列: {hf_dataset.column_names}")
    print(f"  第一条: input_ids 长度={len(hf_dataset[0]['input_ids'])}")

    # 3. 可以继续用 HF 的功能
    print(f"\n  features: {hf_dataset.features}")
    print(f"  可以 push_to_hub、save_to_disk、map 等")

    # 4. 构建 DatasetDict 并保存
    dataset_dict = DatasetDict({"train": hf_dataset})
    dataset_dict.save_to_disk("./local_data/from_tensor_dataset")
    print(f"\n  已保存到 ./local_data/from_tensor_dataset")


# ============================================================
# 五、PyTorch 自定义 Dataset → HuggingFace Dataset
# ============================================================

class MyTorchDataset(TorchDataset):
    """模拟一个自定义 PyTorch Dataset"""

    def __init__(self, num_samples=100):
        self.data = [
            {"text": f"This is sample {i}", "label": i % 2}
            for i in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def demo_custom_torch_to_hf():
    print("\n" + "=" * 60)
    print("五、PyTorch 自定义 Dataset → HuggingFace Dataset")
    print("=" * 60)

    torch_dataset = MyTorchDataset(num_samples=50)
    print(f"PyTorch Dataset 大小: {len(torch_dataset)}")

    # 方法 1：遍历收集所有数据
    all_texts = []
    all_labels = []
    for i in range(len(torch_dataset)):
        item = torch_dataset[i]
        all_texts.append(item["text"])
        all_labels.append(item["label"])

    hf_dataset = Dataset.from_dict({
        "text": all_texts,
        "label": all_labels,
    })
    print(f"\nHF Dataset: {hf_dataset}")
    print(f"  列: {hf_dataset.column_names}")

    # 方法 2：更简洁，直接用列表推导
    items = [torch_dataset[i] for i in range(len(torch_dataset))]
    hf_dataset2 = Dataset.from_list(items)
    print(f"\nfrom_list 方式: {hf_dataset2}")

    # 转换后可以使用 HF 生态的所有功能
    print("\n转换后可用 HF 功能:")
    print("  dataset.map(tokenize_fn)    — 批量分词")
    print("  dataset.filter(...)         — 过滤数据")
    print("  dataset.train_test_split()  — 切分")
    print("  dataset.push_to_hub(...)    — 推送到 Hub")
    print("  dataset.save_to_disk(...)   — 保存到本地")


# ============================================================
# 六、完整流程：双向转换 + Trainer 训练
# ============================================================

def demo_full_pipeline():
    print("\n" + "=" * 60)
    print("六、完整流程：HF Dataset → TensorDataset → 训练 → HF Dataset")
    print("=" * 60)

    # Step 1: 从 HF Hub 加载
    hf_dataset = load_dataset("imdb", split="train[:200]")
    print(f"Step 1: 从 HF Hub 加载 {len(hf_dataset)} 条")

    # Step 2: 分词
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    print(f"Step 2: 分词完成，列: {tokenized.column_names}")

    # Step 3: 转为 TensorDataset
    tensor_ds = TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        tokenized["label"],
    )
    print(f"Step 3: 转为 TensorDataset，大小: {len(tensor_ds)}")

    # Step 4: 用 PyTorch DataLoader 训练
    from torch.utils.data import random_split
    import torch.nn as nn

    train_ds, val_ds = random_split(tensor_ds, [160, 40])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # 简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(30522, 64)
            self.fc = nn.Sequential(
                nn.Linear(64 * 64, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )

        def forward(self, input_ids, attention_mask):
            # 简化演示：embedding + flatten + fc
            emb = self.embedding(input_ids)  # [B, 64, 64]
            # 用 attention_mask 加权
            mask = attention_mask.unsqueeze(-1).float()
            emb = (emb * mask).sum(dim=1) / mask.sum(dim=1)  # [B, 64]
            return self.fc(emb)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_x, batch_mask, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x, batch_mask)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Step 5: 收集预测结果，转回 HF Dataset
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in val_loader:
            out = model(batch_x, batch_mask)
            preds = torch.argmax(out, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.tolist())

    results_ds = Dataset.from_dict({
        "prediction": all_preds,
        "label": all_labels,
    })
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nStep 5: 预测结果转回 HF Dataset")
    print(f"  验证集准确率: {accuracy:.4f}")
    print(f"  HF Dataset: {results_ds}")
    print(f"  features: {results_ds.features}")


# ============================================================
# 运行所有演示
# ============================================================

if __name__ == "__main__":
    demo_hf_to_torch_dataloader()
    # demo_hf_to_torch_custom()
    # demo_hf_to_tensor_dataset()
    # demo_tensor_to_hf()
    # demo_custom_torch_to_hf()
    # demo_full_pipeline()
