"""
BERT 问答系统和命名实体识别

本文件介绍 BERT 在序列标注任务上的应用：
1. 抽取式问答（Extractive QA）
2. 命名实体识别（NER）
3. Token 分类任务
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# 1. 抽取式问答（Extractive QA）
# ============================================================
"""
抽取式问答任务：
- 输入：问题 + 上下文
- 输出：上下文中的答案片段（start 和 end 位置）

BERT 架构：
- 输入：[CLS] question [SEP] context [SEP]
- 输出：每个 token 作为答案开始/结束位置的概率
"""

def qa_demo():
    """问答系统演示"""
    
    print("=" * 60)
    print("抽取式问答系统演示")
    print("=" * 60)
    
    # 使用预训练问答 pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased"
    )
    
    # 注意：实际问答模型应该使用专门的 QA 模型
    # 这里演示的是基础 BERT，实际应该用 distilbert-base-cased-distilled-squad 等
    
    # 使用专门的 QA 模型
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )
    
    # 问答示例
    context = """
    BERT is a transformer-based model that was pre-trained on a large corpus of text.
    It was introduced by Google researchers in 2018. BERT uses masked language modeling
    and next sentence prediction as its pre-training tasks. The model can be fine-tuned
    on various downstream tasks like classification, question answering, and more.
    """
    
    questions = [
        "What is BERT?",
        "Who introduced BERT?",
        "When was BERT introduced?",
        "What are the pre-training tasks of BERT?",
    ]
    
    print("\n上下文:")
    print(context[:100] + "...")
    
    print("\n问答结果:")
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"\n  问题: {question}")
        print(f"  答案: {result['answer']}")
        print(f"  置信度: {result['score']:.4f}")
    
    return qa_pipeline


# ============================================================
# 2. 问答模型微调
# ============================================================

def qa_finetuning_demo():
    """问答模型微调演示"""
    
    print("\n" + "=" * 60)
    print("问答模型微调演示")
    print("=" * 60)
    
    # 加载 SQuAD 数据集
    dataset = load_dataset("squad")
    
    # 取子集
    small_train = dataset["train"].shuffle(seed=42).select(range(1000))
    small_validation = dataset["validation"].shuffle(seed=42).select(range(200))
    
    print(f"\n训练集大小: {len(small_train)}")
    print(f"验证集大小: {len(small_validation)}")
    
    # 查看示例
    example = small_train[0]
    print(f"\n示例:")
    print(f"  问题: {example['question']}")
    print(f"  上下文: {example['context'][:100]}...")
    print(f"  答案: {example['answers']}")
    
    # 加载 Tokenizer 和模型
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    # 数据预处理函数
    def prepare_train_features(examples):
        # 对问题和上下文进行编码
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=384,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # 获取答案位置
        start_positions = []
        end_positions = []
        
        for i, answers in enumerate(examples["answers"]):
            # 取第一个答案
            answer = answers["text"][0] if answers["text"] else ""
            answer_start = answers["answer_start"][0] if answers["answer_start"] else 0
            
            # 找到答案在编码后的位置
            # 简化处理：实际需要考虑 offset_mapping
            start_positions.append(0)  # 简化
            end_positions.append(0)    # 简化
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        
        return inputs
    
    # 应用预处理
    print("\n正在预处理数据...")
    tokenized_train = small_train.map(
        prepare_train_features,
        batched=True,
        remove_columns=small_train.column_names
    )
    
    print(f"预处理完成: {len(tokenized_train)} 条")
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir="./bert-qa",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
    )
    
    print("\n问答模型微调配置完成")
    print("（实际训练需要更多数据和计算资源）")
    
    return trainer


# ============================================================
# 3. 命名实体识别（NER）
# ============================================================
"""
命名实体识别任务：
- 输入：文本序列
- 输出：每个 token 的实体标签

常见实体类型：
- PER: 人名
- ORG: 组织名
- LOC: 地名
- MISC: 其他

标签格式：
- B-PER: 人名开始
- I-PER: 人名内部
- O: 非实体
"""

def ner_demo():
    """命名实体识别演示"""
    
    print("\n" + "=" * 60)
    print("命名实体识别演示")
    print("=" * 60)
    
    # 使用预训练 NER pipeline
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )
    
    # 测试文本
    text = "My name is Sarah and I live in London. I work at Google."
    
    print(f"\n输入文本: {text}")
    
    # 执行 NER
    entities = ner_pipeline(text)
    
    print("\n识别到的实体:")
    for entity in entities:
        print(f"  {entity['word']:20s} -> {entity['entity_group']:6s} (置信度: {entity['score']:.4f})")
    
    # 更多示例
    test_texts = [
        "Apple is looking at buying U.K. startup for $1 billion",
        "Barack Obama was the 44th President of the United States.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    
    print("\n更多示例:")
    for text in test_texts:
        print(f"\n  文本: {text}")
        entities = ner_pipeline(text)
        for entity in entities:
            print(f"    {entity['word']:20s} -> {entity['entity_group']}")
    
    return ner_pipeline


# ============================================================
# 4. NER 模型微调
# ============================================================

def ner_finetuning_demo():
    """NER 模型微调演示"""
    
    print("\n" + "=" * 60)
    print("NER 模型微调演示")
    print("=" * 60)
    
    # 加载 CoNLL2003 NER 数据集
    dataset = load_dataset("conll2003")
    
    # 取子集
    small_train = dataset["train"].shuffle(seed=42).select(range(500))
    small_validation = dataset["validation"].shuffle(seed=42).select(range(100))
    
    print(f"\n训练集大小: {len(small_train)}")
    print(f"验证集大小: {len(small_validation)}")
    
    # 查看标签
    print(f"\n标签列表: {dataset['train'].features['ner_tags'].feature.names}")
    
    # 查看示例
    example = small_train[0]
    print(f"\n示例:")
    print(f"  词: {example['tokens'][:10]}")
    print(f"  标签: {example['ner_tags'][:10]}")
    
    # 加载 Tokenizer 和模型
    model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 获取标签数量
    label_list = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)
    
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # 创建标签映射
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # 数据预处理函数
    def tokenize_and_align_labels(examples):
        # 分词
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            # 对齐标签（处理子词）
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # 特殊 token
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # 子词
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # 应用预处理
    print("\n正在预处理数据...")
    tokenized_train = small_train.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=small_train.column_names
    )
    
    print(f"预处理完成: {len(tokenized_train)} 条")
    
    # 评估指标
    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        
        # 展平并过滤 -100
        true_predictions = [
            p for pred, label in zip(predictions, labels)
            for p, l in zip(pred, label) if l != -100
        ]
        true_labels = [
            l for pred, label in zip(predictions, labels)
            for p, l in zip(pred, label) if l != -100
        ]
        
        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='micro'
        )
        acc = accuracy_score(true_labels, true_predictions)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir="./bert-ner",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_train,  # 简化演示
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    print("\nNER 模型微调配置完成")
    print("（实际训练需要更多数据和计算资源）")
    
    return trainer


# ============================================================
# 5. Token 分类任务
# ============================================================

def token_classification_demo():
    """Token 分类任务演示"""
    
    print("\n" + "=" * 60)
    print("Token 分类任务演示")
    print("=" * 60)
    
    # 加载模型
    model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=9  # CoNLL2003 有 9 个标签
    )
    
    # 测试文本
    text = "Hugging Face Inc. is a company based in New York City."
    
    # 分词
    inputs = tokenizer(text, return_tensors="pt")
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取预测
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    print(f"\n输入文本: {text}")
    print(f"\nToken 分类结果:")
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for token, pred in zip(tokens, predictions[0]):
        print(f"  {token:15s} -> 标签 {pred.item()}")
    
    print("\n说明: 需要训练后才能得到正确的 NER 标签")
    
    return outputs


# ============================================================
# 6. 运行所有演示
# ============================================================

def main():
    """主函数"""
    
    print("BERT 问答系统和命名实体识别教程")
    print("=" * 60)
    
    # 1. 问答系统演示
    qa_demo()
    
    # 2. 问答模型微调演示
    qa_finetuning_demo()
    
    # 3. NER 演示
    ner_demo()
    
    # 4. NER 微调演示
    ner_finetuning_demo()
    
    # 5. Token 分类演示
    token_classification_demo()
    
    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. 抽取式 QA：预测答案在上下文中的开始/结束位置")
    print("  2. NER：为每个 token 分配实体标签")
    print("  3. 使用 BertForQuestionAnswering 和 BertForTokenClassification")
    print("  4. 注意处理子词对齐问题")
    print("\n下一步学习:")
    print("  - bert_04_advanced.py: 高级主题（蒸馏、压缩、多语言）")


if __name__ == "__main__":
    main()
