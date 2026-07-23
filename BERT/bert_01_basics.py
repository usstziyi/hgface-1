"""
BERT 基础入门 - 架构、预训练、Tokenizer

本文件介绍 BERT 的核心概念：
1. BERT 架构原理（Transformer Encoder）
2. 预训练任务（MLM 和 NSP）
3. Tokenizer 使用
4. 模型加载与基础推理
"""

import torch
from transformers import BertTokenizer, BertModel, BertConfig

# ============================================================
# 1. BERT 架构简介
# ============================================================
"""
BERT (Bidirectional Encoder Representations from Transformers) 核心特点：

- 双向上下文：同时利用左右文信息（区别于 GPT 的单向）
- 基于 Transformer Encoder 堆叠
- 预训练 + 微调范式

架构参数（BERT-base）：
- 12 层 Transformer Encoder
- 768 隐藏层维度
- 12 个注意力头
- 1.1 亿参数

BERT-large：
- 24 层
- 1024 维度
- 16 个注意力头
- 3.4 亿参数
"""

# ============================================================
# 2. Tokenizer 使用
# ============================================================

def tokenizer_demo():
    """演示 BERT Tokenizer 的使用"""
    
    # 加载预训练 Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    print("=" * 60)
    print("BERT Tokenizer 演示")
    print("=" * 60)
    
    # 基础分词
    text = "Hello, I am learning BERT!"
    print(f"\n原始文本: {text}")
    
    # 方法1：直接分词
    tokens = tokenizer.tokenize(text)
    print(f"分词结果: {tokens}")
    
    # 方法2：转换为模型输入
    inputs = tokenizer(text, return_tensors="pt")
    print(f"\n模型输入:")
    print(f"  input_ids: {inputs['input_ids']}")
    print(f"  attention_mask: {inputs['attention_mask']}")
    print(f"  token_type_ids: {inputs['token_type_ids']}")
    
    # 解码回文本
    decoded = tokenizer.decode(inputs['input_ids'][0])
    print(f"\n解码结果: {decoded}")
    
    # 特殊 Token
    print("\n特殊 Token:")
    print(f"  [CLS] token id: {tokenizer.cls_token_id}")
    print(f"  [SEP] token id: {tokenizer.sep_token_id}")
    print(f"  [PAD] token id: {tokenizer.pad_token_id}")
    print(f"  [UNK] token id: {tokenizer.unk_token_id}")
    
    return tokenizer


# ============================================================
# 3. 模型加载与基础推理
# ============================================================

def model_inference_demo():
    """演示 BERT 模型的基础推理"""
    
    print("\n" + "=" * 60)
    print("BERT 模型推理演示")
    print("=" * 60)
    
    # 加载模型和 Tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # 准备输入
    text = "BERT is a powerful language model."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    print(f"\n输入文本: {text}")
    print(f"输入 shape: {inputs['input_ids'].shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 输出解析
    last_hidden_states = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    
    print(f"\n输出:")
    print(f"  last_hidden_state shape: {last_hidden_states.shape}")
    print(f"  pooler_output shape: {pooler_output.shape}")
    
    # 解释输出
    print("\n输出说明:")
    print("  - last_hidden_state: 每个 token 的隐藏层表示 [batch_size, seq_len, hidden_dim]")
    print("  - pooler_output: [CLS] token 的表示，经过线性层和 tanh 激活 [batch_size, hidden_dim]")
    
    # 获取 [CLS] token 表示（常用于分类任务）
    cls_token = last_hidden_states[:, 0, :]
    print(f"\n[CLS] token 表示 shape: {cls_token.shape}")
    
    return model, tokenizer


# ============================================================
# 4. 批量处理
# ============================================================

def batch_processing_demo():
    """演示批量文本处理"""
    
    print("\n" + "=" * 60)
    print("批量处理演示")
    print("=" * 60)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    # 多条文本
    texts = [
        "Hello, how are you?",
        "BERT is amazing!",
        "I love natural language processing.",
    ]
    
    print(f"\n输入文本数量: {len(texts)}")
    
    # 批量编码
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,        # 填充到相同长度
        truncation=True,     # 截断超长文本
        max_length=128       # 最大长度
    )
    
    print(f"批量输入 shape:")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  attention_mask: {inputs['attention_mask'].shape}")
    
    # 批量推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\n批量输出:")
    print(f"  last_hidden_state: {outputs.last_hidden_state.shape}")
    print(f"  pooler_output: {outputs.pooler_output.shape}")
    
    # 获取每条文本的 [CLS] 表示
    sentence_embeddings = outputs.pooler_output
    print(f"\n句子嵌入 shape: {sentence_embeddings.shape}")
    
    return sentence_embeddings


# ============================================================
# 5. 句子相似度计算
# ============================================================

def similarity_demo():
    """演示使用 BERT 计算句子相似度"""
    
    print("\n" + "=" * 60)
    print("句子相似度计算演示")
    print("=" * 60)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    # 句子对
    sentences = [
        "The cat sits on the mat",
        "A cat is sitting on a mat",
        "The dog runs in the park",
        "Machine learning is fascinating",
    ]
    
    print("\n句子列表:")
    for i, sent in enumerate(sentences):
        print(f"  {i}: {sent}")
    
    # 编码所有句子
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取句子嵌入
    embeddings = outputs.pooler_output
    
    # 计算余弦相似度
    print("\n余弦相似度矩阵:")
    print("    ", "  ".join([f"S{i}" for i in range(len(sentences))]))
    
    for i in range(len(sentences)):
        similarities = []
        for j in range(len(sentences)):
            # 余弦相似度公式
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()
            similarities.append(sim)
        print(f"S{i}  {['%.3f' % s for s in similarities]}")
    
    print("\n说明: S0 和 S1 语义相似，相似度较高；S0/S1 与 S2/S3 差异较大")


# ============================================================
# 6. 自定义配置
# ============================================================

def custom_config_demo():
    """演示自定义 BERT 配置"""
    
    print("\n" + "=" * 60)
    print("自定义配置演示")
    print("=" * 60)
    
    # 从预训练配置加载
    config = BertConfig.from_pretrained("bert-base-uncased")
    print(f"\nBERT-base 配置:")
    print(f"  隐藏层维度: {config.hidden_size}")
    print(f"  层数: {config.num_hidden_layers}")
    print(f"  注意力头数: {config.num_attention_heads}")
    print(f"  中间层维度: {config.intermediate_size}")
    print(f"  词表大小: {config.vocab_size}")
    
    # 自定义配置（小型 BERT）
    small_config = BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        vocab_size=30522,
    )
    
    print(f"\n自定义小型 BERT 配置:")
    print(f"  隐藏层维度: {small_config.hidden_size}")
    print(f"  层数: {small_config.num_hidden_layers}")
    print(f"  注意力头数: {small_config.num_attention_heads}")
    
    # 随机初始化模型（不加载预训练权重）
    small_model = BertModel(small_config)
    param_count = sum(p.numel() for p in small_model.parameters())
    print(f"\n小型 BERT 参数量: {param_count:,}")
    
    # 对比预训练模型
    base_model = BertModel.from_pretrained("bert-base-uncased")
    base_param_count = sum(p.numel() for p in base_model.parameters())
    print(f"BERT-base 参数量: {base_param_count:,}")


# ============================================================
# 7. 运行所有演示
# ============================================================

if __name__ == "__main__":
    print("BERT 基础入门教程")
    print("=" * 60)
    
    # 1. Tokenizer 演示
    tokenizer_demo()
    
    # 2. 模型推理演示
    model_inference_demo()
    
    # 3. 批量处理演示
    batch_processing_demo()
    
    # 4. 相似度计算演示
    similarity_demo()
    
    # 5. 自定义配置演示
    custom_config_demo()
    
    print("\n" + "=" * 60)
    print("基础教程完成！")
    print("=" * 60)
    print("\n下一步学习:")
    print("  - bert_02_classification.py: 文本分类微调")
    print("  - bert_03_qa_ner.py: 问答系统和命名实体识别")
    print("  - bert_04_advanced.py: 高级主题")
