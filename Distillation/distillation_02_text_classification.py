"""
模型蒸馏学习 2：文本分类蒸馏 — BERT → 小模型
难度：⭐⭐ 中等

将 BERT 大模型蒸馏为小型 Transformer 模型。
使用 HuggingFace 的 Trainer 实现完整的蒸馏流程。

场景：情感分析（IMDb 数据集）
教师：bert-base-uncased (110M 参数)
学生：自定义小型 Transformer (约 5M 参数)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np

# ============================================================
# 一、准备数据和教师模型
# ============================================================

def prepare_teacher_and_data():
    print("=" * 60)
    print("一、准备教师模型和数据")
    print("=" * 60)

    # 加载教师模型
    teacher_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(teacher_name)
    teacher = BertForSequenceClassification.from_pretrained(
        teacher_name,
        num_labels=2,  # 二分类：正面/负面
    )
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"教师模型: {teacher_name}")
    print(f"教师参数量: {teacher_params / 1e6:.1f}M")
    
    # 加载数据
    dataset = load_dataset("imdb")
    print(f"\n数据集: {dataset}")
    
    # 取子集
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    
    print(f"训练集: {len(train_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    # 预处理
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # 教师推理，获取软标签
    print("\n教师模型推理，生成软标签...")
    teacher.eval()
    
    def add_teacher_logits(examples):
        inputs = {
            "input_ids": torch.tensor(examples["input_ids"]),
            "attention_mask": torch.tensor(examples["attention_mask"]),
        }
        with torch.no_grad():
            outputs = teacher(**inputs)
        examples["teacher_logits"] = outputs.logits.tolist()
        return examples
    
    train_dataset = train_dataset.map(add_teacher_logits, batched=True)
    test_dataset = test_dataset.map(add_teacher_logits, batched=True)
    
    # 设置格式
    train_dataset.set_format(type="torch")
    test_dataset.set_format(type="torch")
    
    print(f"教师软标签已添加到数据集")
    
    return tokenizer, teacher, train_dataset, test_dataset


# ============================================================
# 二、定义学生模型
# ============================================================

class SmallTransformer(nn.Module):
    """小型 Transformer 分类模型"""
    
    def __init__(self, vocab_size, num_labels, hidden_size=128, num_layers=2, num_heads=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(512, hidden_size)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # Transformer 编码
        if attention_mask is not None:
            # 将 padding 位置 mask 掉
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # 使用 [CLS] token (第一个 token) 的表示
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits


def create_student_model(vocab_size, num_labels=2):
    """创建学生模型"""
    student = SmallTransformer(
        vocab_size=vocab_size,
        num_labels=num_labels,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )
    
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\n学生模型:")
    print(f"  隐藏层维度: 128")
    print(f"  层数: 2")
    print(f"  注意力头数: 2")
    print(f"  参数量: {student_params / 1e6:.2f}M")
    
    return student


# ============================================================
# 三、蒸馏训练器
# ============================================================

class DistillationTrainer(Trainer):
    """自定义蒸馏训练器"""
    
    def __init__(self, teacher_model=None, temperature=5.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 学生前向传播
        student_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        # 硬标签损失
        labels = inputs.get("labels")
        if labels is not None:
            loss_hard = F.cross_entropy(student_logits, labels)
        else:
            loss_hard = 0
        
        # 软标签损失（蒸馏）
        if self.teacher_model is not None and "teacher_logits" in inputs:
            teacher_logits = inputs["teacher_logits"]
            
            # 温度缩放
            T = self.temperature
            
            # KL 散度
            loss_soft = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            
            # 总损失
            loss = self.alpha * loss_soft + (1 - self.alpha) * loss_hard
        else:
            loss = loss_hard
        
        return (loss, student_logits) if return_outputs else loss


# ============================================================
# 四、训练和评估
# ============================================================

def demo_distillation_training():
    print("\n" + "=" * 60)
    print("二、蒸馏训练")
    print("=" * 60)

    # 准备数据和教师
    tokenizer, teacher, train_dataset, test_dataset = prepare_teacher_and_data()
    
    # 创建学生模型
    student = create_student_model(vocab_size=tokenizer.vocab_size)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./distilled_student",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
    )
    
    # 评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
    
    # 创建蒸馏训练器
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=5.0,
        alpha=0.7,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 训练
    print("\n开始蒸馏训练...")
    print(f"温度 T: 5.0")
    print(f"蒸馏权重 α: 0.7")
    
    train_result = trainer.train()
    
    print(f"\n训练结果:")
    print(f"  训练损失: {train_result.metrics['train_loss']:.4f}")
    
    # 评估
    eval_result = trainer.evaluate()
    print(f"\n评估结果:")
    print(f"  验证准确率: {eval_result['eval_accuracy']:.4f}")
    
    # 保存学生模型
    trainer.save_model("./distilled_student/best")
    print("\n学生模型已保存到 ./distilled_student/best")
    
    # 对比
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n模型对比:")
    print(f"  教师: {teacher_params / 1e6:.1f}M 参数")
    print(f"  学生: {student_params / 1e6:.2f}M 参数")
    print(f"  压缩比: {teacher_params / student_params:.1f}x")


# ============================================================
# 五、对比：蒸馏 vs 直接训练
# ============================================================

def demo_distillation_vs_direct():
    print("\n" + "=" * 60)
    print("三、蒸馏 vs 直接训练对比")
    print("=" * 60)

    print("""
    蒸馏的优势：
    
    ┌─────────────────┬──────────────┬──────────────┐
    │ 训练方式         │ 准确率       │ 说明          │
    ├─────────────────┼──────────────┼──────────────┤
    │ 直接训练学生     │ ~75%         │ 只学习硬标签  │
    │ 蒸馏训练学生     │ ~82%         │ 学习教师知识  │
    │ 教师模型         │ ~92%         │ 大模型上限    │
    └─────────────────┴──────────────┴──────────────┘
    
    蒸馏让学生模型：
    - 从教师的软标签中学到"暗知识"
    - 理解类别间的相似性关系
    - 在相同参数量下达到更高准确率
    
    实际应用：
    - MobileBERT: BERT 的蒸馏版本，用于移动端
    - DistilBERT: 保留 97% 性能，参数量减少 40%
    - TinyBERT: 更小的 BERT 蒸馏版本
    """)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_distillation_training()
    demo_distillation_vs_direct()
