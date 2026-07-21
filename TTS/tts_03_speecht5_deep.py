"""
TTS 学习 3：SpeechT5 深入 — 架构、推理、语音克隆
难度：⭐⭐⭐ 中高

SpeechT5 是微软提出的统一语音处理框架。
预训练任务同时涵盖语音识别 (ASR) 和语音合成 (TTS)。

论文: "SpeechT5: Unified-Modal Encoder-Decoder Framework for Speech"
GitHub: https://github.com/microsoft/SpeechT5

核心架构：
- 共享的 Encoder-Decoder 框架
- 预训练：掩码预测（类似 BERT）
- 微调：TTS、ASR、语音转换等
"""

import torch
import numpy as np

# ============================================================
# 一、SpeechT5 架构详解
# ============================================================

def demo_speecht5_architecture():
    print("=" * 60)
    print("一、SpeechT5 架构详解")
    print("=" * 60)

    from transformers import SpeechT5ForTextToSpeech

    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    print("SpeechT5 TTS 模型结构:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params / 1e6:.2f}M 参数")

    config = model.config
    print(f"\n关键配置:")
    print(f"  词表大小: {config.vocab_size}")
    print(f"  隐藏层维度: {config.hidden_size}")
    print(f"  编码器层数: {config.encoder_layers}")
    print(f"  解码器层数: {config.decoder_layers}")
    print(f"  注意力头数: {config.encoder_attention_heads}")
    print(f"  Mel bin 数: {config.num_mel_bins}")
    print(f"  Speaker embedding 维度: {config.speaker_embedding_dim}")

    print("\nSpeechT5 架构特点:")
    print("  1. Encoder: 处理文本输入（字符级别）")
    print("  2. Decoder: 自回归生成 Mel 声谱图")
    print("  3. Speaker Embedding: 控制音色")
    print("  4. 预训练在 60k 小时语音数据上")

    print("\n统一框架：")
    print("  ┌──────────────────────────────────────┐")
    print("  │     SpeechT5 (共享 Encoder-Decoder)   │")
    print("  │                                       │")
    print("  │  预训练: 掩码预测 (Masked Prediction) │")
    print("  │                                       │")
    print("  │  微调:                                │")
    print("  │  ├─ TTS: 文本 → Mel 声谱图            │")
    print("  │  ├─ ASR: 语音 → 文本                  │")
    print("  │  ├─ ST: 语音翻译                      │")
    print("  │  └─ VC: 语音转换                      │")
    print("  └──────────────────────────────────────┘")


# ============================================================
# 二、Speaker Embeddings 与语音克隆
# ============================================================

def demo_speaker_embeddings():
    print("\n" + "=" * 60)
    print("二、Speaker Embeddings 与语音克隆")
    print("=" * 60)

    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    text = "This is a test of speaker embedding control."
    inputs = processor(text=text, return_tensors="pt")

    print(f"输入文本: {text}")
    print(f"Speaker embedding 维度: 512")

    # 方法 1：零向量（默认音色）
    print("\n--- 方法 1: 零向量（默认音色）---")
    speaker_emb_zero = torch.zeros(1, 512)
    with torch.no_grad():
        speech_zero = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_emb_zero,
            vocoder=vocoder,
        )
    print(f"  输出长度: {len(speech_zero)} 采样点")
    print(f"  时长: {len(speech_zero) / 16000:.2f} 秒")

    # 方法 2：随机向量（随机音色）
    print("\n--- 方法 2: 随机向量（随机音色）---")
    torch.manual_seed(42)
    speaker_emb_random = torch.randn(1, 512)
    with torch.no_grad():
        speech_random = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_emb_random,
            vocoder=vocoder,
        )
    print(f"  输出长度: {len(speech_random)} 采样点")

    # 方法 3：从参考音频提取（真正的语音克隆）
    print("\n--- 方法 3: 从参考音频提取 Speaker Embedding ---")
    print("  需要一个预训练的 speaker encoder（如 x-vector）")
    print("  步骤:")
    print("    1. 加载参考音频（目标说话人）")
    print("    2. 用 speaker encoder 提取 embedding")
    print("    3. 将 embedding 传给 SpeechT5")

    # 演示使用 HuggingFace 的 speaker encoder
    try:
        from transformers import SpeechT5ForSpeechToSpeech  # 或其他模型

        # 使用 SpeechBrain 的 x-vector 模型
        # pip install speechbrain
        print("\n  推荐: 使用 SpeechBrain 的 ECAPA-TDNN")
        print("  from speechbrain.inference.speaker import EncoderClassifier")
        print("  classifier = EncoderClassifier.from_hparams(")
        print("      source='speechbrain/spkrec-ecapa-voxceleb'")
        print("  )")
        print("  embedding = classifier.encode_batch(waveform)")

    except ImportError:
        print("\n  安装 SpeechBrain: pip install speechbrain")

    # 保存音频
    try:
        import soundfile as sf
        sf.write("./output_speecht5_zero.wav", speech_zero.numpy(), samplerate=16000)
        sf.write("./output_speecht5_random.wav", speech_random.numpy(), samplerate=16000)
        print("\n已保存音频文件")
    except ImportError:
        pass


# ============================================================
# 三、批量推理
# ============================================================

def demo_batch_inference():
    print("\n" + "=" * 60)
    print("三、批量推理")
    print("=" * 60)

    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    texts = [
        "Hello, how are you today?",
        "The weather is nice and sunny.",
        "I love machine learning and natural language processing.",
    ]

    speaker_embeddings = torch.zeros(1, 512)

    print(f"文本数量: {len(texts)}")

    # 逐条生成（SpeechT5 目前不支持 batch generate）
    for i, text in enumerate(texts):
        inputs = processor(text=text, return_tensors="pt")

        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder,
            )

        duration = len(speech) / 16000
        print(f"  [{i+1}] '{text[:40]}...' → {duration:.2f} 秒")

    print("\n注意:")
    print("  - SpeechT5 是自回归模型，逐帧生成")
    print("  - 不支持 batch 推理（每次只能处理一条）")
    print("  - 如需批量处理，考虑 FastSpeech2 或 VITS")


# ============================================================
# 四、控制语速和韵律
# ============================================================

def demo_prosody_control():
    print("\n" + "=" * 60)
    print("四、控制语速和韵律")
    print("=" * 60)

    print("""
    SpeechT5 的韵律控制：
    
    1. 文本层面控制：
       - 使用标点符号控制停顿
       - 逗号: 短停顿
       - 句号: 长停顿
       - 省略号: 更长停顿
    
    2. Speaker Embedding 影响：
       - 不同的 embedding 可能有不同的语速
       - 这是隐式的控制方式
    
    3. 更精细的控制需要：
       - FastSpeech2: 直接控制 duration, pitch, energy
       - 或使用专门的韵律控制模型
    
    文本技巧示例：
    
    正常: "Hello, how are you?"
    强调: "Hello! How are YOU?"
    缓慢: "Hello... how... are... you?"
    快速: "Hello, how are you?" (短句)
    """)

    # 演示不同标点的影响
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    texts = [
        "Hello, how are you?",
        "Hello! How are you!",
        "Hello... how... are... you?",
    ]

    speaker_embeddings = torch.zeros(1, 512)

    print("\n不同标点对输出长度的影响:")
    for text in texts:
        inputs = processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder,
            )
        duration = len(speech) / 16000
        print(f"  '{text}' → {duration:.2f} 秒")


# ============================================================
# 五、SpeechT5 的局限性和替代方案
# ============================================================

def demo_limitations():
    print("\n" + "=" * 60)
    print("五、SpeechT5 的局限性和替代方案")
    print("=" * 60)

    limitations = """
    SpeechT5 局限性：
    
    1. 只支持英文
       - 预训练数据只有英文
       - 中文需要其他模型（MMS, XTTS）
    
    2. Speaker Embedding 需要外部提供
       - 没有内置的 speaker encoder
       - 需要额外的模型（如 SpeechBrain）
    
    3. 自回归生成速度较慢
       - 逐帧生成 Mel 声谱图
       - 不适合实时场景
    
    4. 音质上限有限
       - 16kHz 采样率
       - 不如 Bark (24kHz) 或 XTTS
    
    替代方案：
    
    需求                    推荐方案
    ─────────────────────────────────
    中文 TTS                MMS (facebook/mms-tts-cmn)
    高质量英文              Bark / XTTS
    语音克隆                XTTS / SpeechT5 + Speaker Encoder
    实时 TTS                FastSpeech2 / VITS
    多语言                  MMS / Bark
    情感控制                Bark
    """
    print(limitations)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_speecht5_architecture()
    demo_speaker_embeddings()
    demo_batch_inference()
    demo_prosody_control()
    demo_limitations()
