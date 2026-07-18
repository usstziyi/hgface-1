"""
项目 2：大模型指令微调 — 使用 LoRA 微调 flan-t5-base 实现对话摘要生成
使用 PEFT 库进行参数高效微调，Seq2SeqTrainer 训练。
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# 1. 准备数据
# SAMSum 数据集：包含人工编写的对话及其摘要，常用于对话摘要生成任务
dataset = load_dataset("knkarthick/samsum")
train_data = dataset["train"].select(range(1000))  # 取子集

def format_example(example):
    # 添加 "summarize: " 前缀是为了告诉模型这是一个摘要任务，这是 flan-t5 的指令格式
    # 输出: {'text': 'summarize: Alice: Hi, how are you?\nBob: I'm good, thanks!', 'summary': 'Alice and Bob greet each other.'}
    return {"text": f"summarize: {example['dialogue']}", "summary": example["summary"]}

train_data = train_data.map(format_example)

# 2. 加载模型与分词器
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 3. 数据分词
def tokenize_function(examples):
    # 对输入文本进行分词，返回包含 input_ids 和 attention_mask 的字典
    inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )
    # 对目标摘要进行分词，返回包含 input_ids 和 attention_mask 的字典
    targets = tokenizer(
        examples["summary"], padding="max_length", truncation=True, max_length=128
    )
    # inputs 包含：input_ids, attention_mask, labels（labels 来自 targets 的 input_ids）
    inputs["labels"] = targets["input_ids"]
    return inputs


tokenized_train = train_data.map(tokenize_function, batched=True)

# 4. 配置 LoRA 并包装模型
lora_config = LoraConfig(
    r=8,                              # LoRA 的秩（rank），控制低秩矩阵的维度，值越小参数量越少
    lora_alpha=32,                    # LoRA 的缩放参数，用于控制低秩矩阵对原始权重的影响程度
    target_modules=["q", "v"],        # 指定应用 LoRA 的模块，这里选择注意力层的查询（q）和值（v）投影矩阵
    lora_dropout=0.05,                # LoRA 层的 dropout 概率，用于防止过拟合
    bias="none",                      # 是否训练偏置项，"none" 表示不训练任何偏置
    task_type=TaskType.SEQ_2_SEQ_LM,  # 任务类型，指定为序列到序列语言模型
)
# 使用 LoRA 配置包装基础模型，创建 PEFT 模型
# get_peft_model 会将 LoRA 层注入到指定的 target_modules 中
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # 查看可训练参数量

# 5. 训练
training_args = Seq2SeqTrainingArguments(
    output_dir="./lora-flan-t5",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    eval_strategy="no",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    processing_class=tokenizer,
)
trainer.train()

# 6. 保存 LoRA adapter
peft_model.save_pretrained("./my-lora-adapter")

# 加载 LoRA adapter
# 加载基础模型：首先加载原始的 flan-t5-base 模型，不包含任何微调参数
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# 加载 LoRA adapter：将之前保存的 LoRA 适配器参数加载到基础模型上
# PeftModel.from_pretrained 会将 LoRA 层与基础模型权重合并，形成完整的微调后模型
loaded_model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")

# 7. 推理测试
input_text = "summarize: The meeting was long and discussed Q3 results..."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = loaded_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
