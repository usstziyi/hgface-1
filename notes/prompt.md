## Negative Prompt 的作用

**Negative Prompt（负面提示词）是用来告诉模型"不要生成什么"的**，相当于给模型划定禁区。

---

## 直观理解

```
只用 Positive Prompt：
"a photo of a cat sitting on a couch"

模型会自由发挥 → 可能生成：
✓ 一只猫在沙发上
✗ 但背景模糊
✗ 猫的脸有点歪
✗ 画面有噪点
```

```
加上 Negative Prompt：
negative_prompt = "blurry, low quality, distorted"

模型会主动避免 → 更可能生成：
✓ 一只猫在沙发上
✓ 画面清晰锐利
✓ 猫脸端正
✓ 高质量输出
```

---

## 为什么需要 Negative Prompt？

### Stable Diffusion 的训练数据问题

```
训练数据包含：
├── 高清精美图片（少数）
├── 普通质量图片（多数）
├── 模糊照片
├── 低分辨率截图
├── 水印/文字覆盖
└── 奇怪的构图

模型不知道你想要"最好的"，它只是按统计规律生成
如果不约束，它会随机从这些质量层级中采样
```

Negative Prompt 本质是**在采样过程中减去你不想要的特征向量**：

```
生成方向 = 朝着 "cat on couch" 走 - 远离 "blurry, low quality"
         = 在高清区域里找猫和沙发
```

---

## 常见 Negative Prompt 模板

```python
# 基础通用（几乎必加）
negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

# 人像专用
negative_prompt = """
bad anatomy, bad hands, missing fingers, extra fingers, 
fused fingers, poorly drawn face, mutated hands, 
deformed, disfigured, ugly, low quality
"""

# 风景/建筑
negative_prompt = """
blurry, low quality, people, text, watermark, 
signature, overexposed, underexposed
"""

# 写实风格
negative_prompt = """
cartoon, anime, painting, illustration, 3d render, 
plastic, smooth, low quality
"""
```

---

## 技术原理（简化）

```python
# 每一步去噪时

# 1. 预测"cat on couch"方向
positive_prediction = unet(latent, text_embedding_of("cat on couch"))

# 2. 预测"blurry"方向
negative_prediction = unet(latent, text_embedding_of("blurry, low quality"))

# 3. 引导采样
final_prediction = negative_prediction + guidance_scale * (positive_prediction - negative_prediction)
#                                                         └── 这个差值就是"往高质量走"
```

`guidance_scale` 越高，向 positive 方向走、远离 negative 方向的力度越大。

---

## 实际效果对比

```python
# 不加 negative prompt
image_without_neg = pipe("a cat sitting on a couch").images[0]
# 结果：可能模糊、构图差、偶尔有奇怪伪影

# 加负面提示词
image_with_neg = pipe(
    "a cat sitting on a couch",
    negative_prompt="blurry, low quality, distorted, ugly"
).images[0]
# 结果：更清晰、更稳定、失败率更低
```

---

## 最佳实践

| 场景 | Positive Prompt | Negative Prompt |
|------|---------------|-----------------|
| 写实照片 | `"a photo of..."` | `"painting, cartoon, 3d, blurry"` |
| 动漫风格 | `"anime style..."` | `"realistic, photo, 3d, blurry"` |
| 产品图 | `"product photo of..."` | `"text, watermark, messy background"` |
| 风景 | `"landscape..."` | `"people, text, watermark"` |

---

## 总结

```
为什么加 Negative Prompt？
→ 训练数据质量参差不齐，模型不知道该避免什么
→ Negative Prompt 明确划出禁区
→ 大幅提升出图质量和稳定性
→ 这是 Stable Diffusion 社区总结出的实战经验

一句话：
Positive Prompt = "我想要什么"
Negative Prompt = "我绝不想要什么"
两者配合，让模型精准命中你的需求
```