"""
BERT 文本分类微调

本文件介绍如何使用 BERT 进行文本分类任务：
1. 微调策略（全量微调 vs 特征提取）
2. 使用 HuggingFace Trainer API 进行微调
3. 情感分析实战案例
4. 模型评估与预测
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# 1. 微调策略说明
# ============================================================
"""
BERT 微调的两种主要方式：

方式1：特征提取（Feature Extraction）
- 冻结 BERT 参数，只训练分类层
- 训练速度快，适合小数据集
- 性能提升有限

方式2：全量微调（Fine-tuning）
- 更新所有 BERT 参数 + 分类层
- 需要更多计算资源
- 性能提升显著，推荐使用

本文件采用全量微调方式。
"""

# ============================================================
# 2. 加载数据集
# ============================================================

def load_sentiment_dataset():
    """加载情感分析数据集（IMDb）"""
    
    print("=" * 60)
    print("加载 IMDb 情感分析数据集")
    print("=" * 60)
    
    # 加载数据集
    dataset = load_dataset("imdb")
    
    # 取子集加速训练（实际使用时可去掉）
    small_train = dataset["train"].shuffle(seed=42).select(range(2000))
    small_test = dataset["test"].shuffle(seed=42).select(range(500))
    
    print(f"\n训练集大小: {len(small_train)}")
    print(f"测试集大小: {len(small_test)}")
    print(f"\n示例数据:")
    print(f"  文本: {small_train[0]['text'][:100]}...")
    print(f"  标签: {small_train[0]['label']} (0=负面, 1=正面)")
    
    return small_train, small_test


# ============================================================
# 3. 数据预处理
# ============================================================

def preprocess_data(train_data, test_data, model_name="bert-base-uncased"):
    """数据预处理：分词和编码"""
    
    print("\n" + "=" * 60)
    print("数据预处理")
    print("=" * 60)
    
    # 加载 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 定义预处理函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    # 批量处理
    print("\n正在分词...")
    tokenized_train = train_data.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing train"
    )
    tokenized_test = test_data.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing test"
    )
    
    print(f"\n分词完成:")
    print(f"  训练集: {len(tokenized_train)} 条")
    print(f"  测试集: {len(tokenized_test)} 条")
    
    return tokenized_train, tokenized_test, tokenizer


# ============================================================
# 4. 加载模型
# ============================================================

def load_classification_model(model_name="bert-base-uncased", num_labels=2):
    """加载用于分类的 BERT 模型"""
    
    print("\n" + "=" * 60)
    print("加载分类模型")
    print("=" * 60)
    
    # 加载预训练模型（带分类头）
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # 查看模型结构
    print(f"\n模型结构:")
    print(f"  基础模型: {model_name}")
    print(f"  分类头: Linear(768 -> {num_labels})")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model


# ============================================================
# 5. 定义评估指标
# ============================================================

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # 计算准确率
    acc = accuracy_score(labels, preds)
    
    # 计算精确率、召回率、F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================
# 6. 配置训练参数
# ============================================================

def setup_training_args():
    """配置训练参数"""
    
    training_args = TrainingArguments(
        output_dir="./bert-sentiment",
        num_train_epochs=3,              # 训练轮数
        per_device_train_batch_size=8,   # 训练批次大小
        per_device_eval_batch_size=16,   # 评估批次大小
        warmup_steps=100,                # 学习率预热步数
        weight_decay=0.01,               # 权重衰减
        learning_rate=2e-5,              # 学习率（微调推荐值）
        logging_dir="./logs",            # 日志目录
        logging_steps=50,                # 每 50 步记录一次
        eval_strategy="epoch",           # 每个 epoch 评估
        save_strategy="epoch",           # 每个 epoch 保存
        load_best_model_at_end=True,     # 训练结束后加载最佳模型
        metric_for_best_model="f1",      # 以 F1 作为最佳模型指标
        greater_is_better=True,          # F1 越大越好
        report_to="none",                # 不报告到外部（如 wandb）
    )
    
    print("\n" + "=" * 60)
    print("训练参数配置")
    print("=" * 60)
    print(f"  训练轮数: {training_args.num_train_epochs}")
    print(f"  批次大小: {training_args.per_device_train_batch_size}")
    print(f"  学习率: {training_args.learning_rate}")
    print(f"  预热步数: {training_args.warmup_steps}")
    print(f"  评估策略: {training_args.eval_strategy}")
    
    return training_args


# ============================================================
# 7. 训练模型
# ============================================================

def train_model(model, train_data, eval_data, training_args):
    """训练模型"""
    
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n训练开始...")
    train_result = trainer.train()
    
    # 打印训练结果
    print("\n训练完成!")
    print(f"  训练损失: {train_result.training_loss:.4f}")
    print(f"  训练时间: {train_result.metrics['train_runtime']:.2f} 秒")
    print(f"  每秒样本数: {train_result.metrics['train_samples_per_second']:.2f}")
    
    # 保存模型
    trainer.save_model("./bert-sentiment-best")
    print("\n最佳模型已保存到: ./bert-sentiment-best")
    
    return trainer


# ============================================================
# 8. 评估模型
# ============================================================

def evaluate_model(trainer):
    """评估模型"""
    
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 执行评估
    eval_results = trainer.evaluate()
    
    print("\n评估结果:")
    for key, value in eval_results.items():
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "")
            print(f"  {metric_name}: {value:.4f}")
    
    return eval_results


# ============================================================
# 9. 预测新文本
# ============================================================

def predict_sentiment(model_path="./bert-sentiment-best"):
    """使用训练好的模型预测新文本"""
    
    print("\n" + "=" * 60)
    print("情感预测")
    print("=" * 60)
    
    # 使用 pipeline 简化推理
    classifier = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path
    )
    
    # 测试文本
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film, complete waste of time. The plot was boring.",
        "An okay movie, nothing special but not bad either.",
        "What an amazing performance! The acting was superb.",
        "I couldn't wait for this movie to end. Very disappointing.",
    ]
    
    print("\n预测结果:")
    for text in test_texts:
        result = classifier(text)[0]
        label = result['label']
        score = result['score']
        print(f"\n  文本: {text[:60]}...")
        print(f"  情感: {label} (置信度: {score:.4f})")
    
    return classifier


# ============================================================
# 10. 特征提取方式（可选）
# ============================================================

def feature_extraction_approach():
    """特征提取方式：冻结 BERT 参数"""
    
    print("\n" + "=" * 60)
    print("特征提取方式（冻结 BERT）")
    print("=" * 60)
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    
    # 冻结 BERT 基础模型的所有参数
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # 只训练分类头
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    print(f"\n说明: 只有分类头的参数会被更新，训练速度快但性能提升有限")
    
    return model


# ============================================================
# 11. 完整训练流程
# ============================================================

def main():
    """完整训练流程"""
    
    print("BERT 文本分类微调教程")
    print("=" * 60)
    
    # 1. 加载数据
    train_data, test_data = load_sentiment_dataset()
    
    # 2. 数据预处理
    tokenized_train, tokenized_test, tokenizer = preprocess_data(train_data, test_data)
    
    # 3. 加载模型
    model = load_classification_model()
    
    # 4. 配置训练参数
    training_args = setup_training_args()
    
    # 5. 训练模型
    trainer = train_model(model, tokenized_train, tokenized_test, training_args)
    
    # 6. 评估模型
    evaluate_model(trainer)
    
    # 7. 预测新文本
    predict_sentiment()
    
    # 8. 展示特征提取方式
    feature_extraction_approach()
    
    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. 全量微调性能优于特征提取")
    print("  2. 学习率推荐 2e-5 到 5e-5")
    print("  3. 使用 Trainer API 简化训练流程")
    print("  4. 小数据集可取子集快速验证")
    print("\n下一步学习:")
    print("  - bert_03_qa_ner.py: 问答系统和命名实体识别")
    print("  - bert_04_advanced.py: 高级主题")


if __name__ == "__main__":
    main()
