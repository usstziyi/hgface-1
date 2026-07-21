"""
TTS 学习 2：HuggingFace TTS 模型入门
难度：⭐⭐ 中等

使用 HuggingFace Transformers 快速上手 TTS：
1. SpeechT5：微软的统一语音模型
2. Bark：Suno AI 的端到端模型
3. MMS：Meta 的多语言 TTS（1000+ 语言）

HuggingFace TTS 模型列表:
https://huggingface.co/models?pipeline_tag=text-to-speech
"""

import torch
import numpy as np

# ============================================================
# 一、SpeechT5：微软的统一语音模型
# ============================================================

def demo_speecht5_basic():
    print("=" * 60)
    print("一、SpeechT5 基础使用")
    print("=" * 60)

    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

    # 加载组件
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    print("SpeechT5 组件:")
    print(f"  处理器: SpeechT5Processor")
    print(f"  声学模型: SpeechT5ForTextToSpeech")
    print(f"  声码器: SpeechT5HifiGan")

    # 统计参数
    model_params = sum(p.numel() for p in model.parameters())
    vocoder_params = sum(p.numel() for p in vocoder.parameters())
    print(f"\n  声学模型参数: {model_params / 1e6:.1f}M")
    print(f"  声码器参数: {vocoder_params / 1e6:.1f}M")

    # 准备文本
    text = "Hello, this is a test of the SpeechT5 text to speech system."
    inputs = processor(text=text, return_tensors="pt")

    print(f"\n输入文本: {text}")
    print(f"输入 shape: {inputs['input_ids'].shape}")

    # 生成语音（需要 speaker embeddings）
    # 使用默认的 speaker embeddings
    speaker_embeddings = torch.zeros(1, 512)

    # 生成 Mel 声谱图
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings,
            vocoder=vocoder,
        )

    print(f"\n输出波形 shape: {speech.shape}")
    print(f"波形范围: [{speech.min():.4f}, {speech.max():.4f}]")

    # 保存音频
    try:
        import soundfile as sf
        sf.write("./output_speecht5.wav", speech.numpy(), samplerate=16000)
        print("已保存: ./output_speecht5.wav")
    except ImportError:
        print("安装 soundfile 可保存音频: pip install soundfile")

    print("\nSpeechT5 特点:")
    print("  - 预训练在大量英文语音数据上")
    print("  - 需要 speaker embeddings 控制音色")
    print("  - 采样率: 16kHz")


# ============================================================
# 二、Bark：Suno AI 的端到端 TTS
# ============================================================

def demo_bark_basic():
    print("\n" + "=" * 60)
    print("二、Bark 基础使用")
    print("=" * 60)

    from transformers import AutoProcessor, AutoModel

    # 加载 Bark
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Bark 参数量: {model_params / 1e6:.1f}M")

    # 基础生成
    text = "Hello, welcome to the Bark text to speech system! [laughs]"
    inputs = processor(text=text)

    print(f"\n输入文本: {text}")
    print(f"输入 keys: {list(inputs.keys())}")

    # 生成语音
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
        )

    # Bark 输出包含多个部分
    print(f"\n输出 shape: {output.shape}")

    # 提取音频（最后一部分是音频 token）
    # Bark 使用 audio codec (EnCodec)
    print("\nBark 特点:")
    print("  - 端到端：文本直接到音频")
    print("  - 支持多语言（13+ 语言）")
    print("  - 支持非语言声音：[laughs], [sighs], [music]")
    print("  - 支持情感控制")
    print("  - 采样率: 24kHz")

    print("\nBark 特殊标记:")
    print("  [laughs]  — 笑声")
    print("  [sighs]   — 叹息")
    print("  [music]   — 音乐")
    print("  [gasps]   — 喘息")
    print("  [clears throat] — 清嗓子")
    print("  ... — 犹豫")
    print("  使用 IT- 前缀指定语言（如 IT- 意大利语）")


# ============================================================
# 三、MMS：Meta 多语言 TTS
# ============================================================

def demo_mms():
    print("\n" + "=" * 60)
    print("三、MMS (Massively Multilingual Speech)")
    print("=" * 60)

    from transformers import VitsModel, AutoTokenizer

    # MMS 支持 1000+ 语言
    # 这里使用英文示例
    model_name = "facebook/mms-tts-eng"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)

    model_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {model_name}")
    print(f"参数量: {model_params / 1e6:.1f}M")

    # 生成语音
    text = "Hello, this is Meta's multilingual speech synthesis."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    print(f"\n输入文本: {text}")
    print(f"输出波形 shape: {output.shape}")

    # 保存
    try:
        import soundfile as sf
        sf.write("./output_mms.wav", output.squeeze().numpy(), samplerate=model.config.sampling_rate)
        print(f"已保存: ./output_mms.wav (采样率: {model.config.sampling_rate})")
    except ImportError:
        print("安装 soundfile 可保存音频: pip install soundfile")

    print("\nMMS 支持的部分语言:")
    languages = {
        "facebook/mms-tts-eng": "英语",
        "facebook/mms-tts-cmn": "中文（普通话）",
        "facebook/mms-tts-spa": "西班牙语",
        "facebook/mms-tts-fra": "法语",
        "facebook/mms-tts-deu": "德语",
        "facebook/mms-tts-jpn": "日语",
        "facebook/mms-tts-kor": "韩语",
        "facebook/mms-tts-ara": "阿拉伯语",
        "facebook/mms-tts-hin": "印地语",
        "facebook/mms-tts-rus": "俄语",
    }
    for model_id, lang in languages.items():
        print(f"  {model_id}: {lang}")

    print("\nMMS 特点:")
    print("  - 基于 VITS 架构")
    print("  - 支持 1100+ 语言")
    print("  - 端到端生成")
    print("  - 开源免费")


# ============================================================
# 四、VITS 模型详解
# ============================================================

def demo_vits_architecture():
    print("\n" + "=" * 60)
    print("四、VITS 模型架构")
    print("=" * 60)

    architecture = """
    VITS (Conditional Variational Autoencoder for TTS):
    
    ┌──────────────────────────────────────────────────────┐
    │                    VITS 架构                          │
    │                                                      │
    │  文本 → ┌─────────────┐ → ┌──────────┐ → 音频       │
    │         │ Text Encoder│   │ Decoder  │              │
    │         │ (Transformer)│   │(HiFi-GAN)│              │
    │         └──────┬──────┘   └─────┬────┘              │
    │                │                │                    │
    │         ┌──────┴──────┐        │                    │
    │         │ Variational │        │                    │
    │         │ Inference   │────────┘                    │
    │         │ (Posterior) │                             │
    │         └─────────────┘                             │
    │                                                      │
    │  关键组件：                                           │
    │  1. Text Encoder: 文本编码                            │
    │  2. Variational Inference: 后验编码器（训练时）        │
    │  3. Normalizing Flows: 归一化流（提升质量）            │
    │  4. Decoder (HiFi-GAN): 声码器                       │
    │  5. Adversarial Training: 对抗训练                    │
    └──────────────────────────────────────────────────────┘
    
    VITS 优势：
    - 端到端训练（不需要单独的声学模型 + 声码器）
    - 生成速度快（非自回归）
    - 音质好（对抗训练）
    - 支持多说话人
    
    论文: "Conditional Variational Autoencoder with Adversarial
           Learning for End-to-End Text-to-Speech" (2021)
    """
    print(architecture)


# ============================================================
# 五、模型对比与选择
# ============================================================

def demo_model_comparison():
    print("\n" + "=" * 60)
    print("五、TTS 模型对比与选择")
    print("=" * 60)

    comparison = """
    HuggingFace TTS 模型对比：
    
    ┌─────────────┬──────────┬──────────┬──────────┬──────────────┐
    │ 模型         │ 参数量    │ 采样率    │ 语言      │ 特点          │
    ├─────────────┼──────────┼──────────┼──────────┼──────────────┤
    │ SpeechT5    │ ~250M    │ 16kHz    │ 英文      │ 统一框架      │
    │             │          │          │          │ 可语音克隆    │
    ├─────────────┼──────────┼──────────┼──────────┼──────────────┤
    │ Bark        │ ~5B      │ 24kHz    │ 13+语言   │ 端到端        │
    │             │          │          │          │ 支持音效      │
    ├─────────────┼──────────┼──────────┼──────────┼──────────────┤
    │ MMS/VITS    │ ~100M    │ 22kHz    │ 1100+语言 │ 多语言        │
    │             │          │          │          │ 轻量级        │
    ├─────────────┼──────────┼──────────┼──────────┼──────────────┤
    │ XTTS        │ ~500M    │ 24kHz    │ 17语言    │ 语音克隆      │
    │             │          │          │          │ 跨语言        │
    ├─────────────┼──────────┼──────────┼──────────┼──────────────┤
    │ FastSpeech2 │ ~50M     │ 22kHz    │ 多语言    │ 快速          │
    │             │          │          │          │ 可控性强      │
    └─────────────┴──────────┴──────────┴──────────┴──────────────┘
    
    选择建议：
    
    场景                    推荐模型
    ─────────────────────────────────
    英文高质量 TTS          SpeechT5 / Bark
    多语言 TTS              MMS (VITS)
    语音克隆                XTTS / SpeechT5
    带情感/音效             Bark
    快速推理                FastSpeech2
    资源受限                MMS (VITS)
    """
    print(comparison)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # 取消注释运行对应示例
    
    # demo_speecht5_basic()
    # demo_bark_basic()
    demo_mms()
    demo_vits_architecture()
    demo_model_comparison()
