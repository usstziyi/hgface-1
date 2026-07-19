"""
ViT 学习 3：ViT 微调与迁移学习
难度：⭐⭐⭐ 中高

将预训练的 ViT 微调到自定义数据集。
涵盖：全量微调、特征提取（冻结 backbone）、LoRA 微调。

数据集：使用 HuggingFace datasets 的 food101（食物分类）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np

# ============================================================
# 一、数据准备
# ============================================================

def prepare_data():
    print("=" * 60)
    print("一、数据准备")
    print("=" * 60)

    # 加载 food101 数据集（101 类食物，每类 1000 张）
    dataset = load_dataset("ethz/food101")
    print(f"数据集: {dataset}")
    print(f"训练集: {len(dataset['train'])}")
    print(f"测试集: {len(dataset['validation'])}")
    print(f"类别数: {dataset['train'].features['label'].num_classes}")

    # 取子集（加速演示）
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    test_dataset = dataset["validation"].shuffle(seed=42).select(range(200))
    
    print(f"\n使用子集:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    # 加载处理器
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)

    # 定义预处理函数
    def preprocess(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    # 使用 map 批量处理
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["image"],
    )
    test_dataset = test_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["image"],
    )

    # 设置 PyTorch 格式
    train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    test_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    print(f"\n预处理后:")
    print(f"  训练集列: {train_dataset.column_names}")
    print(f"  单条样本: {train_dataset[0].keys()}")
    print(f"  pixel_values shape: {train_dataset[0]['pixel_values'].shape}")

    return train_dataset, test_dataset, dataset["train"].features["label"].num_classes


# ============================================================
# 二、全量微调
# ============================================================

def demo_full_finetuning(train_dataset, test_dataset, num_labels):
    print("\n" + "=" * 60)
    print("二、全量微调 ViT")
    print("=" * 60)

    model_name = "google/vit-base-patch16-224"

    # 加载预训练模型，修改分类头
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels, # 类别数
        ignore_mismatched_sizes=True,  # 允许分类头大小不匹配
    )

    print(f"模型: {model_name}")
    print(f"分类头: {model.classifier}")
    # 分类头被重新初始化为 [768, num_labels]

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./vit_finetuned_full",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch", # 每训练完一个 epoch 就在验证集上评估一次
        save_strategy="epoch", # 每训练完一个 epoch 就保存一次模型
        # save_strategy="best", # 仅保存最佳模型
        logging_steps=1,
        load_best_model_at_end=True, # 自动加载最佳模型
        metric_for_best_model="accuracy", # 根据验证集准确率选择最佳模型
    )

    # 定义评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 训练
    print("\n开始训练...")
    train_result = trainer.train()
    
    print(f"\n训练结果:")
    print(f"  训练损失: {train_result.metrics['train_loss']:.4f}")
    print(f"  训练时间: {train_result.metrics['train_runtime']:.1f}s")

    # 评估
    eval_result = trainer.evaluate()
    print(f"\n评估结果:")
    print(f"  验证准确率: {eval_result['eval_accuracy']:.4f}")
    print(f"  验证损失: {eval_result['eval_loss']:.4f}")

    # 保存模型
    trainer.save_model("./vit_finetuned_full/best")
    print("\n模型已保存到 ./vit_finetuned_full/best")

    return model


# ============================================================
# 三、特征提取模式（冻结 backbone）
# ============================================================

def demo_feature_extraction_mode(train_dataset, test_dataset, num_labels):
    print("\n" + "=" * 60)
    print("三、特征提取模式（冻结 backbone）")
    print("=" * 60)

    model_name = "google/vit-base-patch16-224"

    # 加载预训练模型
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 只解冻分类头
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"可训练比例: {trainable_params / total_params:.2%}")
    # 只有分类头 (~768 * 101) 是可训练的

    # 训练参数（使用更大的学习率，因为只有分类头在训练）
    training_args = TrainingArguments(
        output_dir="./vit_feature_extraction",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-3,  # 更大的学习率
        eval_strategy="epoch",
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n开始训练（只训练分类头）...")
    train_result = trainer.train()
    
    eval_result = trainer.evaluate()
    print(f"\n评估结果:")
    print(f"  验证准确率: {eval_result['eval_accuracy']:.4f}")

    print("\n特点:")
    print("  - 训练速度快（只更新少量参数）")
    print("  - 内存占用少")
    print("  - 适合数据量小的场景")
    print("  - 精度可能不如全量微调")


# ============================================================
# 四、部分微调（只微调最后几层）
# ============================================================

def demo_partial_finetuning(train_dataset, test_dataset, num_labels):
    print("\n" + "=" * 60)
    print("四、部分微调（微调最后几层）")
    print("=" * 60)

    model_name = "google/vit-base-patch16-224"

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后 3 层 transformer + 分类头
    num_layers_to_unfreeze = 3
    total_layers = model.config.num_hidden_layers  # 12
    
    for i in range(total_layers - num_layers_to_unfreeze, total_layers):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = True

    # 解冻分类头
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"可训练比例: {trainable_params / total_params:.2%}")
    print(f"微调层数: 最后 {num_layers_to_unfreeze} 层 + 分类头")

    training_args = TrainingArguments(
        output_dir="./vit_partial_finetuned",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n开始训练（微调最后几层）...")
    train_result = trainer.train()
    
    eval_result = trainer.evaluate()
    print(f"\n评估结果:")
    print(f"  验证准确率: {eval_result['eval_accuracy']:.4f}")

    print("\n特点:")
    print("  - 平衡训练速度和精度")
    print("  - 高层特征更通用，低层特征更特定")
    print("  - 微调高层可以适应新任务，同时保留低层的通用特征")


# ============================================================
# 五、使用 LoRA 微调 ViT
# ============================================================

def demo_lora_finetuning(train_dataset, test_dataset, num_labels):
    print("\n" + "=" * 60)
    print("五、LoRA 微调 ViT")
    print("=" * 60)

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("需要安装 peft: pip install peft")
        return

    model_name = "google/vit-base-patch16-224"

    # 加载基础模型
    base_model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,  # LoRA 秩
        lora_alpha=32,  # 缩放系数
        target_modules=["query", "value"],  # 对 Q 和 V 应用 LoRA
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.IMAGE_CLASSIFICATION,
    )

    # 创建 LoRA 模型
    model = get_peft_model(base_model, lora_config)
    
    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"可训练参数 (LoRA): {trainable_params / 1e6:.2f}M")
    print(f"可训练比例: {trainable_params / total_params:.4%}")
    # LoRA 只增加很少的参数

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./vit_lora_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        eval_strategy="epoch",
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n开始 LoRA 训练...")
    train_result = trainer.train()
    
    eval_result = trainer.evaluate()
    print(f"\n评估结果:")
    print(f"  验证准确率: {eval_result['eval_accuracy']:.4f}")

    # 保存 LoRA adapter
    model.save_pretrained("./vit_lora_finetuned/adapter")
    print("\nLoRA adapter 已保存到 ./vit_lora_finetuned/adapter")

    print("\nLoRA 优势:")
    print("  - 参数量极少（通常 < 1%）")
    print("  - 训练速度快")
    print("  - 可以为每个任务保存独立的 adapter")
    print("  - 方便切换任务（加载不同 adapter）")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # 准备数据
    train_dataset, test_dataset, num_labels = prepare_data()
    
    # 选择微调方式（取消注释运行）
    
    # 方式 1：全量微调（最慢，精度最高）
    demo_full_finetuning(train_dataset, test_dataset, num_labels)
    
    # 方式 2：特征提取（最快，精度一般）
    # demo_feature_extraction_mode(train_dataset, test_dataset, num_labels)
    
    # 方式 3：部分微调（平衡）
    # demo_partial_finetuning(train_dataset, test_dataset, num_labels)
    
    # 方式 4：LoRA 微调（参数高效）
    # demo_lora_finetuning(train_dataset, test_dataset, num_labels)
    
    print("\n" + "=" * 60)
    print("微调方式对比")
    print("=" * 60)
    print("""
    ┌─────────────┬──────────┬──────────┬──────────┬────────────┐
    │ 方式         │ 可训练参数 │ 训练速度  │ 内存占用  │ 适用场景    │
    ├─────────────┼──────────┼──────────┼──────────┼────────────┤
    │ 全量微调     │ 100%     │ 慢       │ 高       │ 数据充足    │
    │ 特征提取     │ < 1%     │ 快       │ 低       │ 数据极少    │
    │ 部分微调     │ ~30%     │ 中       │ 中       │ 数据适中    │
    │ LoRA        │ < 1%     │ 快       │ 低       │ 多任务切换  │
    └─────────────┴──────────┴──────────┴──────────┴────────────┘
    """)
