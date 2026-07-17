

### 动手实践路径（详细版）

#### 项目 1：文本分类 — 夯实基础

**任务**：对 IMDb 电影评论进行二分类情感分析（正面/负面）。

**核心目标**：手动跑通数据加载、分词、模型构建、训练、评估的完整流程，并使用 `Trainer` 简化训练循环。

**详细步骤**：

1. **加载数据**
   使用 `datasets` 库直接加载 IMDb 数据集，并取子集加速实验。
   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb")
   # 取小规模数据快速验证
   small_train = dataset["train"].shuffle(seed=42).select(range(2000))
   small_test = dataset["test"].shuffle(seed=42).select(range(500))
   ```
2. **数据预处理**
   加载分词器，定义预处理函数，用 `map` 批量处理。
   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

   def tokenize_function(examples):
       return tokenizer(examples["text"], padding="max_length", truncation=True)

   tokenized_train = small_train.map(tokenize_function, batched=True)
   tokenized_test = small_test.map(tokenize_function, batched=True)
   ```
3. **加载模型**
   使用 `AutoModelForSequenceClassification`，指定类别数为 2。
   ```python
   from transformers import AutoModelForSequenceClassification
   model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
   ```
4. **配置训练参数并训练**
   设置 `TrainingArguments`，交给 `Trainer` 执行。
   ```python
   from transformers import TrainingArguments, Trainer
   training_args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       num_train_epochs=3,
       per_device_train_batch_size=8,
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_test,
   )
   trainer.train()
   ```
5. **评估与预测**
   训练完成后评估模型，并对新文本进行推理。
   ```python
   # 评估
   trainer.evaluate()

   # 预测新文本
   import torch
   inputs = tokenizer("This movie is fantastic!", return_tensors="pt")
   with torch.no_grad():
       logits = model(**inputs).logits
   predicted_class = torch.argmax(logits, dim=-1).item()
   print(predicted_class)  # 1 为正面，0 为负面
   ```

***

#### 项目 2：大模型指令微调 — 掌握 LoRA

**任务**：使用 LoRA 高效微调 `google/flan-t5-base`，实现一个简单的摘要生成或指令遵循模型。

**核心目标**：掌握 `PEFT` 库的 `LoraConfig` 配置、模型包装、`Seq2SeqTrainer` 的使用，以及 adapter 的保存与加载。

**详细步骤**：

1. **准备数据**
   使用 `samsum` 对话摘要数据集，并编写格式化函数，为模型添加任务前缀。
   ```python
   from datasets import load_dataset
   dataset = load_dataset("samsum")
   train_data = dataset["train"].select(range(1000))  # 取子集

   def format_example(example):
       return {"text": f"summarize: {example['dialogue']}", "summary": example["summary"]}

   train_data = train_data.map(format_example)
   ```
2. **加载模型与分词器**
   注意 `T5` 是 `seq2seq` 模型，需要使用对应的自动类。
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   model_name = "google/flan-t5-base"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
   ```
3. **数据分词**
   对输入文本和摘要分别进行分词，注意设置 `max_length` 和截断。
   ```python
   def tokenize_function(examples):
       inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
       targets = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=128)
       inputs["labels"] = targets["input_ids"]
       return inputs

   tokenized_train = train_data.map(tokenize_function, batched=True)
   ```
4. **配置 LoRA 并包装模型**
   关键一步，设置 `target_modules`，通常对 `q` 和 `v` 投影层应用 LoRA。
   ```python
   from peft import LoraConfig, get_peft_model, TaskType
   lora_config = LoraConfig(
       r=8,  # 低秩矩阵的秩
       lora_alpha=32,
       target_modules=["q", "v"],  # T5 的查询和值投影层
       lora_dropout=0.05,
       bias="none",
       task_type=TaskType.SEQ_2_SEQ_LM
   )
   peft_model = get_peft_model(model, lora_config)
   peft_model.print_trainable_parameters()  # 查看可训练参数量
   ```
5. **训练**
   使用 `Seq2SeqTrainer` 处理序列到序列任务。
   ```python
   from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
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
   ```
6. **保存与加载 LoRA adapter**
   - **保存**：`peft_model.save_pretrained("./my-lora-adapter")`
   - **加载**：先加载基础模型，再用 `PeftModel` 加载 adapter。
     ```python
     from peft import PeftModel
     base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
     loaded_model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")
     ```
7. **推理测试**
   ```python
   input_text = "summarize: The meeting was long and discussed Q3 results..."
   inputs = tokenizer(input_text, return_tensors="pt")
   outputs = loaded_model.generate(**inputs, max_new_tokens=50)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

***

#### 项目 3：多模态探索 — 图文检索

**任务**：使用 CLIP 模型进行零样本图像分类或图文检索。

**核心目标**：理解多模态模型中，文本和图像表示如何在同一空间对齐，并学会使用 `processor` 统一处理不同模态的输入。

**详细步骤**：

1. **加载模型与处理器**
   使用 `CLIPProcessor` 同时处理文本和图像。
   ```python
   from transformers import CLIPProcessor, CLIPModel
   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
   ```
2. **零样本图像分类**
   给定一张图像和候选类别文本，计算图像与每个文本的相似度。
   ```python
   from PIL import Image
   import requests
   import torch

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
   for label, prob in zip(labels, probs[0]):
       print(f"{label}: {prob:.4f}")
   ```
3. **图文检索（扩展）**
   若有多个图像和描述，可通过计算 `logits_per_image` 和 `logits_per_text` 矩阵实现双向检索。

完成这三个项目后，你将具备从数据处理、模型微调到多模态应用的完整开发能力。每个项目建议先在 Colab 上跑通，然后再替换成自己的数据或模型进行扩展。遇到具体报错随时贴出来，我们共同排查。
