"""
项目 5：使用 load_dataset 加载数据集 & map 批量处理
涵盖：从 Hub 加载、从本地文件加载、map 配合 tokenizer 的各种用法。
"""

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import os
import json
import csv

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ============================================================
# 一、从 Hugging Face Hub 加载数据集
# ============================================================

def demo_hub_dataset():
    print("=" * 60)
    print("一、从 Hugging Face Hub 加载数据集")
    print("=" * 60)

    # 1. 加载完整数据集
    dataset = load_dataset("imdb")
    print(f"类型: {type(dataset)}")          # DatasetDict
    print(f"包含的 split: {dataset.keys()}")  # ['train', 'test']
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"训练集第一条: {dataset['train'][0].keys()}")  # ['text', 'label']

    # 2. 加载指定 split（节省内存）
    train_only = load_dataset("imdb", split="train[:1000]")
    print(f"\n只加载 train 前 1000 条: {len(train_only)}")

    # 3. 加载子集 + 取样本
    small_train = dataset["train"].shuffle(seed=42).select(range(500))
    print(f"取 500 条子集: {len(small_train)}")

    # 4. 查看数据集结构
    print(f"\n数据集特征: {dataset['train'].features}")
    print(f"前 3 条标签: {dataset['train'][:3]['label']}")


# ============================================================
# 二、从本地文件加载数据集
# ============================================================

def demo_local_dataset():
    print("\n" + "=" * 60)
    print("二、从本地文件加载数据集")
    print("=" * 60)

    os.makedirs("./local_data", exist_ok=True)

    # --- 1. CSV 文件 ---
    csv_path = "./local_data/reviews.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows([
            {"text": "This movie is great!", "label": 1},
            {"text": "Terrible acting and plot.", "label": 0},
            {"text": "I loved every minute of it.", "label": 1},
            {"text": "Boring and predictable.", "label": 0},
            {"text": "A masterpiece of modern cinema.", "label": 1},
        ])

    csv_dataset = load_dataset("csv", data_files=csv_path)
    print(f"CSV 数据集: {csv_dataset}")
    print(f"  第一条: {csv_dataset['train'][0]}")

    # --- 2. JSON 文件 ---
    json_path = "./local_data/reviews.json"
    data = [
        {"text": "Absolutely wonderful film!", "label": 1},
        {"text": "Waste of time, do not watch.", "label": 0},
        {"text": "Great story and performances.", "label": 1},
        {"text": "Dull and uninteresting.", "label": 0},
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    json_dataset = load_dataset("json", data_files=json_path)
    print(f"\nJSON 数据集: {json_dataset}")
    print(f"  第一条: {json_dataset['train'][0]}")

    # --- 3. 纯文本文件（每行一条） ---
    text_path = "./local_data/sentences.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Machine learning is transforming the world.\n")
        f.write("Transformers revolutionized NLP.\n")

    text_dataset = load_dataset("text", data_files=text_path)
    print(f"\nText 数据集: {text_dataset}")
    print(f"  第一条: {text_dataset['train'][0]}")  # {'text': '...'}

    # --- 4. 从 Python 字典直接创建 ---
    raw_data = {
        "text": ["Hello world", "How are you", "Nice to meet you"],
        "label": [0, 1, 0],
    }
    dict_dataset = Dataset.from_dict(raw_data)
    print(f"\n从字典创建: {dict_dataset}")

    # --- 5. 从列表创建 ---
    list_data = [
        {"text": "Sample A", "label": 0},
        {"text": "Sample B", "label": 1},
    ]
    list_dataset = Dataset.from_list(list_data)
    print(f"从列表创建: {list_dataset}")


# ============================================================
# 三、map 函数配合 tokenizer 批量处理
# ============================================================

def demo_map_tokenizer():
    print("\n" + "=" * 60)
    print("三、map 函数配合 tokenizer 批量处理")
    print("=" * 60)

    dataset = load_dataset("imdb")
    small_train = dataset["train"].shuffle(seed=42).select(range(200))

    # --- 1. 基础 map：对每条文本分词 ---
    def tokenize_basic(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized = small_train.map(tokenize_basic, batched=True)
    print(f"map 后新增的列: {[c for c in tokenized.column_names if c not in small_train.column_names]}")
    print(f"  第一条 input_ids 长度: {len(tokenized[0]['input_ids'])}")

    # --- 2. 设置 padding + max_length ---
    def tokenize_padded(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized_padded = small_train.map(tokenize_padded, batched=True)
    print(f"\npadding 后 input_ids 长度: {len(tokenized_padded[0]['input_ids'])}")  # 固定 128

    # --- 3. map 中创建新字段 ---
    def add_text_length(examples):
        examples["text_length"] = [len(t) for t in examples["text"]]
        return examples

    with_length = small_train.map(add_text_length, batched=True)
    print(f"\n新增 text_length 字段: {with_length[0]['text_length']}")

    # --- 4. map 中过滤数据 ---
    def filter_short(examples):
        # 只保留文本长度 > 200 的样本
        return [len(t) > 200 for t in examples["text"]]

    filtered = small_train.map(filter_short, batched=True)
    # 注意：上面的 filter 返回的是 bool 列表，但 map 不会自动过滤
    # 需要用 dataset.filter() 方法
    filtered = small_train.filter(lambda x: len(x["text"]) > 200)
    print(f"\n过滤前: {len(small_train)} 条, 过滤后 (text > 200 字符): {len(filtered)} 条")

    # --- 5. 使用 num_proc 多进程加速 ---
    import time

    def slow_tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    t0 = time.time()
    tokenized_single = small_train.map(slow_tokenize, batched=True)
    t1 = time.time()
    print(f"\n单进程 map 耗时: {t1 - t0:.2f}s")

    t0 = time.time()
    tokenized_multi = small_train.map(slow_tokenize, batched=True, num_proc=4)
    t1 = time.time()
    print(f"多进程 map (num_proc=4) 耗时: {t1 - t0:.2f}s")

    # --- 6. 移除不需要的列（减少内存占用） ---
    columns_to_remove = [c for c in small_train.column_names if c != "label"]
    tokenized_clean = small_train.map(
        tokenize_basic,
        batched=True,
        remove_columns=columns_to_remove,
    )
    print(f"\n移除原始列后剩余列: {tokenized_clean.column_names}")

    # --- 7. map 处理 seq2seq 任务（输入 + 标签分别编码） ---
    def tokenize_seq2seq(examples):
        inputs = tokenizer(examples["text"], truncation=True, max_length=256)
        targets = tokenizer(examples["summary"], truncation=True, max_length=64)
        inputs["labels"] = targets["input_ids"]
        return inputs

    # 模拟一个摘要数据集
    summary_data = Dataset.from_dict({
        "text": [f"summarize: This is article {i} about technology." for i in range(50)],
        "summary": [f"Article {i} summary." for i in range(50)],
    })
    tokenized_seq2seq = summary_data.map(tokenize_seq2seq, batched=True, remove_columns=["text", "summary"])
    print(f"\nSeq2Seq 处理后列: {tokenized_seq2seq.column_names}")
    print(f"  包含 labels: {'labels' in tokenized_seq2seq.column_names}")

    # --- 8. map 处理多模态（图像 + 文本） ---
    def tokenize_multimodal(examples):
        # 文本部分用 tokenizer
        text_encodings = tokenizer(examples["caption"], truncation=True, max_length=77)
        # 图像部分通常由 processor 处理，这里演示文本编码
        return text_encodings

    multimodal_data = Dataset.from_dict({
        "caption": [
            "a photo of a cat",
            "a dog running in the park",
            "a beautiful sunset over the ocean",
        ],
        "image_id": [1, 2, 3],
    })
    tokenized_mm = multimodal_data.map(tokenize_multimodal, batched=True)
    print(f"\n多模态文本编码后列: {tokenized_mm.column_names}")


# ============================================================
# 四、数据集操作：合并、切分、保存
# ============================================================

def demo_dataset_operations():
    print("\n" + "=" * 60)
    print("四、数据集操作：合并、切分、保存")
    print("=" * 60)

    # 1. 合并两个数据集
    ds1 = Dataset.from_dict({"text": ["Hello", "World"], "label": [0, 1]})
    ds2 = Dataset.from_dict({"text": ["Foo", "Bar"], "label": [1, 0]})
    merged = concatenate_datasets([ds1, ds2])
    print(f"合并后大小: {len(merged)}")  # 4

    # 2. 切分数据集（train/test split）
    full = Dataset.from_dict({
        "text": [f"Sample {i}" for i in range(100)],
        "label": [i % 2 for i in range(100)],
    })
    split = full.train_test_split(test_size=0.2, seed=42)
    print(f"\n切分后 train: {len(split['train'])}, test: {len(split['test'])}")

    # 3. 构建 DatasetDict
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test": split["test"],
    })
    print(f"DatasetDict keys: {dataset_dict.keys()}")

    # 4. 保存到本地
    save_dir = "./local_data/saved_dataset"
    dataset_dict.save_to_disk(save_dir)
    print(f"\n已保存到 {save_dir}")

    # 5. 从本地加载
    loaded = load_dataset(save_dir)
    print(f"从本地加载: {loaded}")
    print(f"  train 大小: {len(loaded['train'])}")

    # 6. 推送到 Hub（需要登录，仅演示代码）
    # dataset_dict.push_to_hub("your-username/your-dataset-name")
    print("\n推送到 Hub: dataset.push_to_hub('username/dataset-name')")


# ============================================================
# 运行所有演示
# ============================================================

if __name__ == "__main__":
    demo_hub_dataset()
    # demo_local_dataset()
    # demo_map_tokenizer()
    # demo_dataset_operations()
