"""
BERT 高级主题

本文件介绍 BERT 的高级应用和优化技术：
1. 模型蒸馏（DistilBERT、TinyBERT）
2. 模型压缩与量化
3. 多语言 BERT（mBERT、XLM-R）
4. 部署优化（ONNX、TorchScript）
5. 实际应用案例
"""

import torch
import time
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    AutoTokenizer,
    AutoModel,
    pipeline,
)
from transformers import pipeline

# ============================================================
# 1. 模型蒸馏 - DistilBERT
# ============================================================
"""
DistilBERT 是 BERT 的蒸馏版本：
- 参数量减少 40%
- 推理速度提升 60%
- 性能保留 97%

蒸馏方法：
- 使用 BERT-base 作为教师模型
- 训练更小的学生模型（6层 vs 12层）
- 损失函数：软标签 + 隐藏层对齐 + 注意力对齐
"""

def distilbert_comparison():
    """DistilBERT 与 BERT 对比"""
    
    print("=" * 60)
    print("DistilBERT 与 BERT 对比")
    print("=" * 60)
    
    # 加载模型
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 统计参数量
    bert_params = sum(p.numel() for p in bert_model.parameters())
    distilbert_params = sum(p.numel() for p in distilbert_model.parameters())
    
    print(f"\n参数量对比:")
    print(f"  BERT-base: {bert_params:,} (100%)")
    print(f"  DistilBERT: {distilbert_params:,} ({distilbert_params/bert_params*100:.1f}%)")
    print(f"  减少: {(1 - distilbert_params/bert_params)*100:.1f}%")
    
    # 推理速度对比
    text = "This is a test sentence for comparing BERT and DistilBERT performance."
    
    # BERT 推理
    inputs_bert = bert_tokenizer(text, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs_bert = bert_model(**inputs_bert)
    bert_time = time.time() - start
    
    # DistilBERT 推理
    inputs_distil = distilbert_tokenizer(text, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs_distil = distilbert_model(**inputs_distil)
    distil_time = time.time() - start
    
    print(f"\n推理速度对比（单次）:")
    print(f"  BERT-base: {bert_time*1000:.2f} ms")
    print(f"  DistilBERT: {distil_time*1000:.2f} ms")
    print(f"  加速: {bert_time/distil_time:.2f}x")
    
    # 输出维度对比
    print(f"\n输出维度对比:")
    print(f"  BERT-base: {outputs_bert.last_hidden_state.shape}")
    print(f"  DistilBERT: {outputs_distil.last_hidden_state.shape}")
    
    print("\n说明: DistilBERT 保留了相同的隐藏层维度（768），但层数从 12 减少到 6")
    
    return bert_model, distilbert_model


# ============================================================
# 2. TinyBERT - 更小的蒸馏模型
# ============================================================
"""
TinyBERT 是更激进的蒸馏版本：
- 4 层 Transformer
- 312 隐藏层维度
- 参数量仅 14.5M（BERT-base 的 1/7.5）
- 适合移动端部署
"""

def tinybert_demo():
    """TinyBERT 演示"""
    
    print("\n" + "=" * 60)
    print("TinyBERT 演示")
    print("=" * 60)
    
    # 使用 prajjwal1/bert-tiny 作为示例
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 统计参数
    params = sum(p.numel() for p in model.parameters())
    
    print(f"\nTinyBERT 配置:")
    print(f"  模型: {model_name}")
    print(f"  参数量: {params:,}")
    print(f"  层数: {model.config.num_hidden_layers}")
    print(f"  隐藏层维度: {model.config.hidden_size}")
    print(f"  注意力头数: {model.config.num_attention_heads}")
    
    # 推理测试
    text = "TinyBERT is very efficient for mobile deployment."
    inputs = tokenizer(text, return_tensors="pt")
    
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start
    
    print(f"\n推理时间: {inference_time*1000:.2f} ms")
    print(f"输出 shape: {outputs.last_hidden_state.shape}")
    
    print("\n适用场景:")
    print("  - 移动端应用")
    print("  - 嵌入式设备")
    print("  - 实时推理场景")
    print("  - 资源受限环境")
    
    return model


# ============================================================
# 3. 模型量化
# ============================================================
"""
模型量化技术：
- 将 FP32 权重转换为 INT8
- 减少 75% 内存占用
- 提升推理速度 2-4x
- 性能损失通常 < 1%
"""

def quantization_demo():
    """模型量化演示"""
    
    print("\n" + "=" * 60)
    print("模型量化演示")
    print("=" * 60)
    
    # 加载原始模型
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # 原始模型大小
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"\n原始模型:")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  数据类型: FP32")
    print(f"  模型大小: {original_size / (1024**2):.2f} MB")
    
    # 动态量化（PyTorch）
    from torch.quantization import quantize_dynamic
    
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # 量化 Linear 层
        dtype=torch.qint8
    )
    
    # 量化后模型大小（估算）
    quantized_size = original_size * 0.25  # INT8 是 FP32 的 1/4
    
    print(f"\n量化后模型:")
    print(f"  数据类型: INT8")
    print(f"  估算大小: {quantized_size / (1024**2):.2f} MB")
    print(f"  压缩率: {original_size/quantized_size:.2f}x")
    
    # 推理测试
    text = "Quantization reduces model size significantly."
    inputs = tokenizer(text, return_tensors="pt")
    
    # 原始模型推理
    start = time.time()
    with torch.no_grad():
        _ = model(**inputs)
    original_time = time.time() - start
    
    # 量化模型推理
    start = time.time()
    with torch.no_grad():
        _ = quantized_model(**inputs)
    quantized_time = time.time() - start
    
    print(f"\n推理速度对比:")
    print(f"  原始模型: {original_time*1000:.2f} ms")
    print(f"  量化模型: {quantized_time*1000:.2f} ms")
    
    print("\n量化方法:")
    print("  1. 训练后量化（PTQ）: 简单快速，无需重新训练")
    print("  2. 量化感知训练（QAT）: 训练时模拟量化，性能更好")
    print("  3. 动态量化: 推理时动态量化激活值")
    
    return quantized_model


# ============================================================
# 4. 多语言 BERT
# ============================================================
"""
多语言模型：

1. mBERT (Multilingual BERT):
   - 104 种语言
   - 共享词表和参数
   - 支持跨语言迁移

2. XLM-R (XLM-Roberta):
   - 100+ 种语言
   - 更大的训练数据
   - 性能优于 mBERT
"""

def multilingual_bert_demo():
    """多语言 BERT 演示"""
    
    print("\n" + "=" * 60)
    print("多语言 BERT 演示")
    print("=" * 60)
    
    # 加载 mBERT
    mbert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    mbert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    print(f"\nmBERT 配置:")
    print(f"  支持语言: 104")
    print(f"  词表大小: {mbert_model.config.vocab_size}")
    print(f"  参数量: {sum(p.numel() for p in mbert_model.parameters()):,}")
    
    # 多语言测试
    texts = {
        "English": "Hello, how are you?",
        "Chinese": "你好，最近怎么样？",
        "French": "Bonjour, comment allez-vous?",
        "Spanish": "Hola, ¿cómo estás?",
        "Japanese": "こんにちは、お元気ですか？",
    }
    
    print("\n多语言编码测试:")
    for lang, text in texts.items():
        inputs = mbert_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = mbert_model(**inputs)
        embedding = outputs.pooler_output
        print(f"  {lang:10s}: {text[:30]:30s} -> 嵌入维度 {embedding.shape}")
    
    # 跨语言相似度
    print("\n跨语言相似度（'你好' vs 'Hello'）:")
    text1 = "你好"
    text2 = "Hello"
    
    inputs1 = mbert_tokenizer(text1, return_tensors="pt")
    inputs2 = mbert_tokenizer(text2, return_tensors="pt")
    
    with torch.no_grad():
        emb1 = mbert_model(**inputs1).pooler_output
        emb2 = mbert_model(**inputs2).pooler_output
    
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    print(f"  相似度: {similarity:.4f}")
    
    print("\n说明: mBERT 能将不同语言的相似语义映射到相近的向量空间")
    
    return mbert_model


# ============================================================
# 5. XLM-R 演示
# ============================================================

def xlmr_demo():
    """XLM-R 演示"""
    
    print("\n" + "=" * 60)
    print("XLM-R 演示")
    print("=" * 60)
    
    # 加载 XLM-R
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"\nXLM-R 配置:")
    print(f"  支持语言: 100+")
    print(f"  词表大小: {model.config.vocab_size}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试中文
    text = "XLM-R 是一个强大的多语言模型"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\n中文测试:")
    print(f"  输入: {text}")
    print(f"  输出 shape: {outputs.last_hidden_state.shape}")
    
    print("\nXLM-R 优势:")
    print("  - 更大的训练语料（2.5TB  Common Crawl）")
    print("  - 更好的跨语言迁移能力")
    print("  - 在多个基准测试上优于 mBERT")
    
    return model


# ============================================================
# 6. ONNX 导出
# ============================================================
"""
ONNX (Open Neural Network Exchange):
- 跨平台模型格式
- 支持多种推理引擎
- 优化推理性能
"""

def onnx_export_demo():
    """ONNX 导出演示"""
    
    print("\n" + "=" * 60)
    print("ONNX 导出演示")
    print("=" * 60)
    
    # 加载模型
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    # 准备示例输入
    text = "This is a sample text for ONNX export."
    inputs = tokenizer(text, return_tensors="pt")
    
    print(f"\n准备导出模型到 ONNX 格式...")
    print(f"  模型: {model_name}")
    print(f"  输入: {list(inputs.keys())}")
    
    # 导出命令（实际执行需要 torch.onnx）
    print("\n导出命令:")
    print("""
    import torch
    from transformers import BertModel, BertTokenizer
    
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 准备示例输入
    text = "Sample text"
    inputs = tokenizer(text, return_tensors="pt")
    
    # 导出 ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        "bert_model.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "output": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )
    """)
    
    print("\nONNX 推理（使用 ONNX Runtime）:")
    print("""
    import onnxruntime as ort
    
    # 加载 ONNX 模型
    session = ort.InferenceSession("bert_model.onnx")
    
    # 准备输入
    inputs = tokenizer(text, return_tensors="np")
    
    # 推理
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
    })
    """)
    
    print("\nONNX 优势:")
    print("  - 跨平台部署（Python、C++、Java 等）")
    print("  - 硬件加速（GPU、NPU、FPGA）")
    print("  - 推理优化（图优化、算子融合）")
    
    return model


# ============================================================
# 7. TorchScript 导出
# ============================================================

def torchscript_demo():
    """TorchScript 导出演示"""
    
    print("\n" + "=" * 60)
    print("TorchScript 导出演示")
    print("=" * 60)
    
    # 加载模型
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    # 准备示例输入
    text = "This is a sample text for TorchScript export."
    inputs = tokenizer(text, return_tensors="pt")
    
    print(f"\n准备导出模型到 TorchScript 格式...")
    
    # 方法1：Tracing
    print("\n方法1: Tracing")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        )
    
    # 保存
    traced_model.save("bert_traced.pt")
    print("  已保存: bert_traced.pt")
    
    # 加载
    loaded_model = torch.jit.load("bert_traced.pt")
    print("  加载成功")
    
    # 方法2：Scripting
    print("\n方法2: Scripting")
    print("""
    # 需要模型使用 TorchScript 兼容的语法
    scripted_model = torch.jit.script(model)
    scripted_model.save("bert_scripted.pt")
    """)
    
    print("\nTorchScript 优势:")
    print("  - PyTorch 原生支持")
    print("  - 可部署到生产环境（无需 Python）")
    print("  - 支持 C++ 推理")
    print("  - 性能优化（算子融合、内存优化）")
    
    return traced_model


# ============================================================
# 8. 实际应用案例
# ============================================================

def practical_applications():
    """实际应用案例"""
    
    print("\n" + "=" * 60)
    print("实际应用案例")
    print("=" * 60)
    
    # 案例1：文本分类
    print("\n案例1: 情感分析")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, not great.",
    ]
    for text in texts:
        result = classifier(text)[0]
        print(f"  {text:30s} -> {result['label']} ({result['score']:.3f})")
    
    # 案例2：问答系统
    print("\n案例2: 问答系统")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    context = "BERT was created by Google researchers in 2018."
    question = "Who created BERT?"
    result = qa(question=question, context=context)
    print(f"  问题: {question}")
    print(f"  答案: {result['answer']} (置信度: {result['score']:.3f})")
    
    # 案例3：命名实体识别
    print("\n案例3: 命名实体识别")
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    text = "Apple Inc. is based in California."
    entities = ner(text)
    print(f"  文本: {text}")
    for entity in entities:
        print(f"    {entity['word']:20s} -> {entity['entity_group']}")
    
    # 案例4：文本相似度
    print("\n案例4: 文本相似度")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    text1 = "The cat sits on the mat"
    text2 = "A cat is sitting on a mat"
    text3 = "The dog runs in the park"
    
    inputs1 = tokenizer(text1, return_tensors="pt")
    inputs2 = tokenizer(text2, return_tensors="pt")
    inputs3 = tokenizer(text3, return_tensors="pt")
    
    with torch.no_grad():
        emb1 = model(**inputs1).pooler_output
        emb2 = model(**inputs2).pooler_output
        emb3 = model(**inputs3).pooler_output
    
    sim12 = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    sim13 = torch.nn.functional.cosine_similarity(emb1, emb3).item()
    
    print(f"  '{text1}' vs '{text2}': {sim12:.4f}")
    print(f"  '{text1}' vs '{text3}': {sim13:.4f}")
    
    print("\n应用场景总结:")
    print("  1. 客服系统: 情感分析 + 问答")
    print("  2. 搜索引擎: 文本相似度 + 语义检索")
    print("  3. 内容审核: 分类 + NER")
    print("  4. 智能推荐: 文本嵌入 + 相似度计算")
    
    return classifier


# ============================================================
# 9. 性能优化建议
# ============================================================

def optimization_tips():
    """性能优化建议"""
    
    print("\n" + "=" * 60)
    print("性能优化建议")
    print("=" * 60)
    
    print("\n1. 模型选择:")
    print("  - 追求性能: BERT-large, RoBERTa-large")
    print("  - 平衡性能: BERT-base, RoBERTa-base")
    print("  - 轻量级: DistilBERT, TinyBERT")
    print("  - 多语言: mBERT, XLM-R")
    
    print("\n2. 推理优化:")
    print("  - 使用 GPU 加速")
    print("  - 批量处理（batch_size=8-32）")
    print("  - 模型量化（INT8）")
    print("  - ONNX/TorchScript 导出")
    print("  - 使用 TensorRT 进一步优化")
    
    print("\n3. 训练优化:")
    print("  - 混合精度训练（FP16）")
    print("  - 梯度累积（增大有效 batch_size）")
    print("  - 学习率预热（warmup_steps）")
    print("  - 早停策略（early stopping）")
    
    print("\n4. 内存优化:")
    print("  - 梯度检查点（gradient checkpointing）")
    print("  - 减小 batch_size")
    print("  - 使用更小的模型")
    print("  - 模型并行（多 GPU）")
    
    print("\n5. 部署建议:")
    print("  - 生产环境: ONNX + TensorRT")
    print("  - 移动端: TinyBERT + 量化")
    print("  - 边缘设备: 模型蒸馏 + 剪枝")
    print("  - 云服务: 容器化 + 自动扩缩容")
    
    print("\n6. 常见问题:")
    print("  - OOM: 减小 batch_size, 使用梯度检查点")
    print("  - 过拟合: 增加数据, 使用 dropout, 早停")
    print("  - 推理慢: 量化, ONNX, 批量处理")
    print("  - 效果差: 调整学习率, 增加训练轮数, 数据清洗")


# ============================================================
# 10. 运行所有演示
# ============================================================

def main():
    """主函数"""
    
    print("BERT 高级主题教程")
    print("=" * 60)
    
    # 1. DistilBERT 对比
    distilbert_comparison()
    
    # 2. TinyBERT 演示
    tinybert_demo()
    
    # 3. 量化演示
    quantization_demo()
    
    # 4. 多语言 BERT
    multilingual_bert_demo()
    
    # 5. XLM-R 演示
    xlmr_demo()
    
    # 6. ONNX 导出
    onnx_export_demo()
    
    # 7. TorchScript 导出
    torchscript_demo()
    
    # 8. 实际应用案例
    practical_applications()
    
    # 9. 优化建议
    optimization_tips()
    
    print("\n" + "=" * 60)
    print("高级教程完成！")
    print("=" * 60)
    print("\n学习总结:")
    print("  1. 模型蒸馏: DistilBERT 减少 40% 参数，速度提升 60%")
    print("  2. 模型量化: INT8 量化减少 75% 内存，速度提升 2-4x")
    print("  3. 多语言: mBERT/XLM-R 支持 100+ 语言")
    print("  4. 部署优化: ONNX/TorchScript 支持跨平台部署")
    print("  5. 实际应用: 分类、QA、NER、相似度计算")
    print("\n下一步:")
    print("  - 尝试在自己的数据上微调 BERT")
    print("  - 探索模型部署到生产环境")
    print("  - 研究最新的 BERT 变体（ALBERT、ELECTRA 等）")


if __name__ == "__main__":
    main()
