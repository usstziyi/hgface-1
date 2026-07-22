你说得对，**diffusers 属于视觉领域**，但它**不限于视觉**。更准确地说，diffusers 是**扩散模型**的专用库，而扩散模型的应用跨了多个模态。

---

## diffusers 的多模态支持

### 1. **视觉（核心领域）**
```python
# 文生图
from diffusers import StableDiffusionPipeline

# 图生图
from diffusers import StableDiffusionImg2ImgPipeline

# 图像修复
from diffusers import StableDiffusionInpaintPipeline

# 超分辨率
from diffusers import StableDiffusionUpscalePipeline
```

### 2. **音频**
```python
# 文生音频
from diffusers import AudioLDM2Pipeline

# 音频到音频转换
from diffusers import AudioDiffusionPipeline
```

### 3. **视频**
```python
# 文生视频
from diffusers import StableVideoDiffusionPipeline

# 视频帧插值
from diffusers import VideoToVideoPipeline
```

### 4. **3D**
```python
# 文生3D
from diffusers import ShapEPipeline  # 生成3D模型
```

---

## 为什么独立出来而不是留在 transformers 的视觉 Pipeline 中？

关键在于**生成方式不同**：

```
transformers 视觉 Pipeline：
输入 → 模型 → 直接输出（分类、检测等）

diffusers Pipeline：
输入 → 文本编码器 → UNet去噪（多步迭代）→ VAE解码 → 输出
      └── 数十到数百步循环 ──┘
```

扩散模型的**迭代去噪**过程完全不同于标准 Transformer 的一次前向传播，所以才需要独立的库来管理采样循环、调度器等组件。

---

## 更准确的归类

```
Hugging Face 生态
├── transformers（通用模型）
│   ├── NLP Pipeline（文本）
│   ├── 视觉 Pipeline（分类、检测、分割）← 判别式视觉
│   └── 语音 Pipeline（识别、合成）
│
└── diffusers（扩散模型专用）
    ├── 视觉生成（文生图、编辑）← 生成式视觉
    ├── 音频生成
    ├── 视频生成
    └── 3D生成
```

所以 **diffusers 是生成式 AI 的独立库**，以视觉为主但不限于视觉。它从 transformers 独立出来不是因为"不属于视觉"，而是因为**扩散模型的范式与传统 Transformer 推理差异太大**，需要专门的管理框架。