"""
项目 2：大模型指令微调 — 使用 LoRA 微调 flan-t5-base 实现对话摘要生成
使用 PEFT 库进行参数高效微调，Seq2SeqTrainer 训练。
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# 1. 准备数据
dataset = load_dataset("samsum")
train_data = dataset["train"].select(range(1000))  # 取子集


def format_example(example):
    return {"text": f"summarize: {example['dialogue']}", "summary": example["summary"]}


train_data = train_data.map(format_example)

# 2. 加载模型与分词器
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 3. 数据分词
def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )
    targets = tokenizer(
        examples["summary"], padding="max_length", truncation=True, max_length=128
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


tokenized_train = train_data.map(tokenize_function, batched=True)

# 4. 配置 LoRA 并包装模型
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=32,
    target_modules=["q", "v"],  # T5 的查询和值投影层
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # 查看可训练参数量

# 5. 训练
training_args = Seq2SeqTrainingArguments(
    output_dir="./lora-flan-t5",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
)
trainer.train()

# 6. 保存 LoRA adapter
peft_model.save_pretrained("./my-lora-adapter")

# 加载 LoRA adapter
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
loaded_model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")

# 7. 推理测试
input_text = "summarize: The meeting was long and discussed Q3 results..."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = loaded_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
