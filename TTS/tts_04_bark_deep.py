"""
TTS 学习 4：Bark 模型深入 — 多语言、情感、音效
难度：⭐⭐⭐⭐ 进阶

Bark 是 Suno AI 开发的端到端 TTS 模型。
特点：支持多语言、情感表达、非语言声音（笑声、音乐等）。

GitHub: https://github.com/suno-ai/bark
论文: 无正式论文，但有技术博客

架构：
- 基于 GPT 风格的自回归 Transformer
- 使用 EnCodec 作为音频 tokenizer
- 支持语义 token + 声学 token 的两阶段生成
"""

import torch
import numpy as np

# ============================================================
# 一、Bark 架构详解
# ============================================================

def demo_bark_architecture():
    print("=" * 60)
    print("一、Bark 架构详解")
    print("=" * 60)

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")

    print("Bark 模型组件:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"  {name}: {params / 1e6:.1f}M 参数")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e9:.2f}B")

    print("\nBark 生成流程:")
    print("""
    ┌──────────────────────────────────────────────────────┐
    │                  Bark 生成流程                        │
    │                                                      │
    │  文本 → ┌──────────────┐ → 语义 tokens              │
    │         │  coarse model │   (内容信息)               │
    │         │  (GPT-style)  │                            │
    │         └──────┬───────┘                             │
    │                │                                     │
    │                ↓                                     │
    │         ┌──────────────┐ → 声学 tokens              │
    │         │  fine model   │   (声学细节)               │
    │         │  (GPT-style)  │                            │
    │         └──────┬───────┘                             │
    │                │                                     │
    │                ↓                                     │
    │         ┌──────────────┐ → 波形                      │
    │         │  EnCodec      │   (24kHz 音频)             │
    │         │  (解码器)     │                             │
    │         └──────────────┘                             │
    └──────────────────────────────────────────────────────┘
    """)

    print("Bark 模型组成:")
    print("  1. Coarse model: 生成语义 token（内容）")
    print("  2. Fine model: 生成声学 token（细节）")
    print("  3. EnCodec: 将 token 解码为波形")


# ============================================================
# 二、多语言支持
# ============================================================

def demo_multilingual():
    print("\n" + "=" * 60)
    print("二、多语言支持")
    print("=" * 60)

    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")

    # 多语言文本
    multilingual_texts = {
        "英语": "Hello, this is Bark speaking English.",
        "中文": "你好，这是 Bark 在说中文。",
        "日语": "こんにちは、Barkが日本語を話しています。",
        "法语": "Bonjour, c'est Bark qui parle français.",
        "德语": "Hallo, das ist Bark, der Deutsch spricht.",
        "西班牙语": "Hola, soy Bark hablando en español.",
    }

    print("Bark 支持的语言:")
    for lang, text in multilingual_texts.items():
        print(f"  {lang}: {text}")

    # 生成示例（英语）
    text = multilingual_texts["英语"]
    inputs = processor(text=text)

    print(f"\n生成示例:")
    print(f"  文本: {text}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
        )

    print(f"  输出 shape: {output.shape}")

    print("\n语言提示前缀（可选）:")
    print("  [EN] — 英语")
    print("  [ZH] — 中文")
    print("  [JA] — 日语")
    print("  [FR] — 法语")
    print("  [DE] — 德语")
    print("  [ES] — 西班牙语")
    print("  [PT] — 葡萄牙语")
    print("  [RU] — 俄语")
    print("  [TR] — 土耳其语")
    print("  [PL] — 波兰语")
    print("  ... 更多语言")

    print("\n注意:")
    print("  - Bark 会自动检测语言")
    print("  - 可以使用前缀强制指定语言")
    print("  - 中文效果可能不如英文")


# ============================================================
# 三、情感和音效控制
# ============================================================

def demo_emotion_and_effects():
    print("\n" + "=" * 60)
    print("三、情感和音效控制")
    print("=" * 60)

    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")

    # 情感表达
    emotion_texts = [
        "Hello! [laughs] That's so funny!",
        "I'm really happy today! [laughs]",
        "Well... [sighs] I'm not sure about that.",
        "Oh my god! [gasps] I can't believe it!",
        "Let me think... [clears throat] ...yes, I agree.",
    ]

    print("情感标记示例:")
    for text in emotion_texts:
        print(f"  {text}")

    # 音乐生成
    music_text = """
    ♪ [music] 
    A beautiful melody playing softly in the background
    ♪
    """

    print(f"\n音乐生成:")
    print(f"  {music_text.strip()}")

    # 生成示例
    text = "Hello! [laughs] That's really funny!"
    inputs = processor(text=text)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
        )

    print(f"\n生成结果 shape: {output.shape}")

    print("\nBark 特殊标记完整列表:")
    print("  [laughs]       — 笑声")
    print("  [sighs]        — 叹息")
    print("  [music]        — 音乐")
    print("  [gasps]        — 喘息")
    print("  [clears throat] — 清嗓子")
    print("  ...            — 犹豫/停顿")
    print("  CAPITAL LETTERS — 强调/大声")
    print("  ♪              — 音乐标记")

    print("\n语音历史预设（Voice Presets）:")
    print("  Bark 支持预定义的说话人风格:")
    print("  v2/en_speaker_0 — en_speaker_9")
    print("  使用方式:")
    print("    history_prompt='v2/en_speaker_0'")


# ============================================================
# 四、Voice Presets（说话人预设）
# ============================================================

def demo_voice_presets():
    print("\n" + "=" * 60)
    print("四、Voice Presets（说话人预设）")
    print("=" * 60)

    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")

    text = "Hello, this is a test of different voice presets."

    # 不同的 speaker presets
    presets = [
        "v2/en_speaker_0",  # 男性
        "v2/en_speaker_3",  # 女性
        "v2/en_speaker_6",  # 男性
        "v2/en_speaker_9",  # 女性
    ]

    print(f"文本: {text}")
    print(f"\n不同 Voice Preset 的生成:")

    for preset in presets:
        inputs = processor(
            text=text,
            voice_preset=preset,
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
            )

        print(f"  {preset}: 输出 shape = {output.shape}")

    print("\n可用的 Voice Presets:")
    print("  英语: v2/en_speaker_0 到 v2/en_speaker_9")
    print("  中文: v2/zh_speaker_0 到 v2/zh_speaker_9")
    print("  日语: v2/ja_speaker_0 到 v2/ja_speaker_9")
    print("  法语: v2/fr_speaker_0 到 v2/fr_speaker_9")
    print("  ... 更多语言")

    print("\n使用方式:")
    print("  inputs = processor(")
    print("      text='Hello!',")
    print("      voice_preset='v2/en_speaker_0',")
    print("  )")


# ============================================================
# 五、生成参数调优
# ============================================================

def demo_generation_params():
    print("\n" + "=" * 60)
    print("五、生成参数调优")
    print("=" * 60)

    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")

    text = "This is a test of generation parameters."

    print(f"文本: {text}")
    print("\n参数对比:")

    # 不同 temperature
    print("\n--- Temperature 对比 ---")
    for temp in [0.4, 0.7, 1.0]:
        inputs = processor(text=text)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=temp,
            )
        print(f"  temperature={temp}: shape={output.shape}")

    print("\nTemperature 说明:")
    print("  低 (0.1-0.4): 更确定，更一致，但可能单调")
    print("  中 (0.5-0.8): 平衡（推荐）")
    print("  高 (0.9-1.0): 更多样，更有表现力，但可能不稳定")

    # 其他参数
    print("\n其他重要参数:")
    print("  do_sample=True     — 启用采样（推荐）")
    print("  min_eos_p=0.2      — 控制停止概率")
    print("  max_length=10000   — 最大生成长度")

    print("\n性能优化:")
    print("  - 使用 GPU 加速")
    print("  - 使用 half precision (float16)")
    print("  - 使用 enable_cpu_offload() 减少显存")
    print("""
    # GPU 优化示例
    model = AutoModel.from_pretrained(
        "suno/bark",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    # CPU offload（显存不足时）
    model.enable_cpu_offload()
    """)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_bark_architecture()
    demo_multilingual()
    demo_emotion_and_effects()
    demo_voice_presets()
    demo_generation_params()
