"""
项目 1：文本分类 — IMDb 电影评论情感分析（正面/负面）
使用 BERT + Trainer 完成数据加载、分词、训练、评估与推理的完整流程。
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# 1. 加载数据
dataset = load_dataset("stanfordnlp/imdb")
# 取小规模数据快速验证
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

# 2. 数据预处理
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = small_train.map(tokenize_function, batched=True)
tokenized_test = small_test.map(tokenize_function, batched=True)

# 3. 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 4. 配置训练参数并训练
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", # 每训练完一个 epoch 就在验证集上评估一次
    num_train_epochs=3,
    per_device_train_batch_size=8,
)
# 每个设备（GPU/CPU）每步训练用 8 条样本。2000 条数据，每步 8 条 → 每个 epoch 250 步

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)
trainer.train()

# 5. 评估与预测
trainer.evaluate()

# 预测新文本
inputs = tokenizer("This movie is fantastic!", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = torch.argmax(logits, dim=-1).item()
print(f"预测类别: {predicted_class}")  # 1 为正面，0 为负面
