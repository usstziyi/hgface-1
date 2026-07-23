"""
TTS 学习 1：音频基础 — 波形、声谱图、音频处理
难度：⭐ 基础

文字转语音 (Text-to-Speech, TTS) 的核心概念：
- 音频信号：波形 (waveform) 和声谱图 (spectrogram)
- 采样率：每秒采样次数（如 16kHz, 22050Hz, 44100Hz）
- 声谱图：音频的频率表示，是 TTS 模型的中间表示

TTS 的基本流程：
  文本 → 声谱图 (声学模型) → 波形 (声码器) → 音频文件
"""

import numpy as np
import torch

# ============================================================
# 一、音频波形基础
# ============================================================

def demo_waveform_basics():
    print("=" * 60)
    print("一、音频波形基础")
    print("=" * 60)

    # 采样率：每秒采样次数
    sample_rate = 16000  # 16kHz，语音常用
    duration = 2.0       # 2 秒

    # 生成简单的正弦波（纯音）
    frequency = 440  # A4 音符，440Hz
    # endpoint=False: 不包含结束值 duration，确保采样点数精确为 sample_rate * duration
    # 这样每个采样点间隔为 1/sample_rate，最后一个采样点在 duration - 1/sample_rate 处
    # 如果 endpoint=True，会包含 duration，导致采样点数多一个或间隔略有不同
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)

    print(f"采样率: {sample_rate} Hz")
    print(f"时长: {duration} 秒")
    print(f"波形长度: {len(waveform)} 个采样点")
    print(f"波形范围: [{waveform.min():.3f}, {waveform.max():.3f}]")
    print(f"频率: {frequency} Hz (A4 音符)")

    # 多频率叠加（模拟更复杂的声音）
    print("\n--- 多频率叠加 ---")
    waveform_complex = (
        0.3 * np.sin(2 * np.pi * 440 * t) +   # A4
        0.2 * np.sin(2 * np.pi * 554 * t) +   # C#5
        0.1 * np.sin(2 * np.pi * 659 * t)     # E5
    )
    print(f"叠加 3 个频率: 440Hz + 554Hz + 659Hz")
    print(f"复合波形范围: [{waveform_complex.min():.3f}, {waveform_complex.max():.3f}]")

    # 采样率对比
    print("\n--- 不同采样率 ---")
    sample_rates = [8000, 16000, 22050, 44100, 48000]
    for sr in sample_rates:
        num_samples = int(sr * duration)
        # 奈奎斯特频率 = sr / 2（能表示的最高频率）
        nyquist = sr / 2
        print(f"  {sr:>6} Hz: {num_samples:>6} 采样点, 最高频率 {nyquist:.0f} Hz")

    print("\n采样率选择:")
    print("  8kHz: 电话语音")
    print("  16kHz: 语音识别、TTS（常用）")
    print("  22050Hz: 一般 TTS 输出")
    print("  44100Hz: CD 音质")
    print("  48kHz: 专业音频")


# ============================================================
# 二、声谱图 (Spectrogram)
# ============================================================

def demo_spectrogram():
    print("\n" + "=" * 60)
    print("二、声谱图 (Spectrogram)")
    print("=" * 60)

    print("""
    声谱图是 TTS 的核心表示：
    
    波形 (Waveform):
    - 时域表示：振幅随时间变化
    - shape: [num_samples]
    
    声谱图 (Spectrogram):
    - 频域表示：频率随时间变化
    - shape: [num_freq_bins, num_frames]
    - X 轴: 时间帧, Y 轴: 频率, 颜色: 能量
    
    ┌────────────────────────────────┐
    │  高频 ┌──┐                     │
    │       │  │                     │
    │  中频 │  ├────┐   ┌──┐        │
    │       │  │    │   │  │        │
    │  低频 └──┘    └───┘  └──      │
    │       t1   t2   t3   t4       │
    └────────────────────────────────┘
    
    Mel 声谱图：
    - 使用 Mel 尺度（模拟人耳感知）
    - 低频分辨率高，高频分辨率低
    - TTS 模型最常用的表示
    """)

    # 使用 PyTorch 计算声谱图
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 生成扫频信号（频率从 100Hz 到 4000Hz）
    freq_start, freq_end = 100, 4000
    freq = freq_start + (freq_end - freq_start) * t / duration
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

    # 短时傅里叶变换 (STFT)
    n_fft = 400       # FFT 窗口大小
    hop_length = 100  # 帧移

    # 使用 torchaudio 的 STFT
    try:
        import torchaudio

        # 声谱图
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
        )
        spectrogram = spec_transform(waveform_tensor)
        print(f"声谱图 shape: {spectrogram.shape}")
        # [num_freq_bins, num_frames]

        # Mel 声谱图
        n_mels = 80
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_spectrogram = mel_transform(waveform_tensor)
        print(f"Mel 声谱图 shape: {mel_spectrogram.shape}")
        # [n_mels, num_frames]

        # 对数 Mel 声谱图（常用）
        log_mel = torch.log(mel_spectrogram + 1e-9)
        print(f"对数 Mel 声谱图 shape: {log_mel.shape}")

        print(f"\n参数说明:")
        print(f"  n_fft: {n_fft} (FFT 窗口大小)")
        print(f"  hop_length: {hop_length} (帧移)")
        print(f"  n_mels: {n_mels} (Mel 滤波器数量)")
        print(f"  帧数: {spectrogram.shape[1]} (时间帧)")
        print(f"  频率 bin 数: {spectrogram.shape[0]}")
        print(f"  Mel bin 数: {mel_spectrogram.shape[0]}")

    except ImportError:
        print("需要安装 torchaudio: pip install torchaudio")
        print("\n手动计算声谱图:")

        # 手动实现 STFT
        window_size = n_fft
        window = torch.hann_window(window_size)

        # 分帧
        num_frames = (len(waveform_tensor) - window_size) // hop_length + 1
        frames = torch.zeros(num_frames, window_size)
        for i in range(num_frames):
            start = i * hop_length
            frames[i] = waveform_tensor[start:start + window_size] * window

        # FFT
        fft_result = torch.fft.rfft(frames, dim=1)
        spectrogram = torch.abs(fft_result).T
        print(f"  声谱图 shape: {spectrogram.shape}")


# ============================================================
# 三、音频文件读写
# ============================================================

def demo_audio_io():
    print("\n" + "=" * 60)
    print("三、音频文件读写")
    print("=" * 60)

    # 方法 1：使用 scipy（无需额外安装）
    print("--- 方法 1: scipy.io.wavfile ---")
    try:
        from scipy.io import wavfile

        # 生成音频
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        # 保存
        wavfile.write("./output_tone.wav", sample_rate, waveform)
        print(f"  已保存: ./output_tone.wav")

        # 读取
        sr, data = wavfile.read("./output_tone.wav")
        print(f"  读取: 采样率={sr}, 长度={len(data)}, dtype={data.dtype}")

    except ImportError:
        print("  需要安装 scipy: pip install scipy")

    # 方法 2：使用 torchaudio
    print("\n--- 方法 2: torchaudio ---")
    try:
        import torchaudio

        # 生成音频（float32，范围 [-1, 1]）
        sample_rate = 22050
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = (0.5 * torch.sin(2 * np.pi * 440 * t)).unsqueeze(0)
        # shape: [1, num_samples] (channels, samples)

        # 保存
        torchaudio.save("./output_tone_torch.wav", waveform, sample_rate)
        print(f"  已保存: ./output_tone_torch.wav")

        # 读取
        data, sr = torchaudio.load("./output_tone_tone.wav")
        print(f"  读取: 采样率={sr}, shape={data.shape}")

    except ImportError:
        print("  需要安装 torchaudio: pip install torchaudio")
    except Exception as e:
        print(f"  注意: {e}")

    # 方法 3：使用 soundfile
    print("\n--- 方法 3: soundfile ---")
    try:
        import soundfile as sf

        # 保存
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

        sf.write("./output_tone_sf.wav", waveform, sample_rate)
        print(f"  已保存: ./output_tone_sf.wav")

        # 读取
        data, sr = sf.read("./output_tone_sf.wav")
        print(f"  读取: 采样率={sr}, 长度={len(data)}")

    except ImportError:
        print("  需要安装 soundfile: pip install soundfile")

    print("\n音频格式:")
    print("  WAV: 无损，文件大，TTS 常用")
    print("  MP3: 有损压缩，文件小")
    print("  FLAC: 无损压缩")
    print("  OGG: 有损压缩，开源")


# ============================================================
# 四、音频特征提取
# ============================================================

def demo_audio_features():
    print("\n" + "=" * 60)
    print("四、音频特征提取")
    print("=" * 60)

    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 生成模拟语音信号（多个频率 + 噪声）
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 500 * t) +
        0.1 * np.sin(2 * np.pi * 1000 * t) +
        0.05 * np.random.randn(len(t))
    )

    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    try:
        import torchaudio

        # 1. MFCC (Mel-Frequency Cepstral Coefficients)
        # 语音识别和 TTS 的重要特征
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
        )
        mfcc = mfcc_transform(signal_tensor)
        print(f"MFCC shape: {mfcc.shape}")
        # [13, num_frames]
        print(f"  13 个 MFCC 系数")
        print(f"  {mfcc.shape[1]} 个时间帧")

        # 2. Mel 声谱图
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        )
        mel_spec = mel_transform(signal_tensor)
        print(f"\nMel 声谱图 shape: {mel_spec.shape}")
        print(f"  80 个 Mel 频率 bin")
        print(f"  {mel_spec.shape[1]} 个时间帧")

        # 3. 音频重采样
        resampler = torchaudio.transforms.Resample(
            orig_freq=16000,
            new_freq=22050,
        )
        resampled = resampler(signal_tensor)
        print(f"\n重采样:")
        print(f"  原始: {len(signal_tensor)} 采样点 (16kHz)")
        print(f"  重采样: {len(resampled)} 采样点 (22050Hz)")

    except ImportError:
        print("需要安装 torchaudio: pip install torchaudio")

    print("\nTTS 中的特征使用:")
    print("  - Mel 声谱图: 声学模型的输出目标")
    print("  - MFCC: 语音识别特征，也可用于 TTS")
    print("  - 声码器: 将 Mel 声谱图转换为波形")


# ============================================================
# 五、TTS 系统架构概览
# ============================================================

def demo_tts_architecture():
    print("\n" + "=" * 60)
    print("五、TTS 系统架构概览")
    print("=" * 60)

    architecture = """
    TTS 系统的基本架构：
    
    两阶段架构（传统）：
    ┌──────┐     ┌──────────────┐     ┌────────┐     ┌──────┐
    │ 文本 │ ──→ │  声学模型     │ ──→ │ 声码器 │ ──→ │ 音频 │
    │      │     │ (Text→Mel)   │     │(Mel→Wav)│    │      │
    └──────┘     └──────────────┘     └────────┘     └──────┘
    
    声学模型：
    - Tacotron2: Seq2Seq + Attention
    - FastSpeech2: 非自回归，更快
    - SpeechT5: 统一的 Seq2Seq 框架
    
    声码器：
    - WaveNet: 自回归，高质量但慢
    - WaveRNN: 更快的自回归声码器
    - HiFi-GAN: 对抗生成，快且高质量
    - Vocoder: 将 Mel 声谱图转为波形
    
    端到端架构（现代）：
    ┌──────┐     ┌──────────────┐     ┌──────┐
    │ 文本 │ ──→ │  端到端模型   │ ──→ │ 音频 │
    │      │     │ (Text→Wav)   │     │      │
    └──────┘     └──────────────┘     └──────┘
    
    端到端模型：
    - VITS: VAE + 归一化流 + 对抗训练
    - Bark: Transformer，支持多语言
    - XTTS: 跨语言语音克隆
    
    主流 TTS 模型对比：
    ┌─────────────┬──────────┬──────────┬──────────────┐
    │ 模型         │ 类型      │ 速度      │ 特点          │
    ├─────────────┼──────────┼──────────┼──────────────┤
    │ Tacotron2   │ 两阶段   │ 中等      │ 经典，质量好  │
    │ FastSpeech2 │ 两阶段   │ 快        │ 非自回归      │
    │ SpeechT5    │ 两阶段   │ 快        │ 统一框架      │
    │ VITS        │ 端到端   │ 快        │ 高质量        │
    │ Bark        │ 端到端   │ 慢        │ 多语言+音效   │
    └─────────────┴──────────┴──────────┴──────────────┘
    """
    print(architecture)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_waveform_basics()
    # demo_spectrogram()
    # demo_audio_io()
    # demo_audio_features()
    # demo_tts_architecture()
