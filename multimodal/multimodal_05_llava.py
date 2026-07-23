"""
多模态学习 5：LLaVA — 多模态大语言模型对话
难度：⭐⭐⭐⭐ 进阶

LLaVA (Large Language and Vision Assistant) 将视觉编码器与 LLM 结合，
实现真正的多模态理解和对话能力。

架构：
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ CLIP ViT    │ ──→ │ 投影层        │ ──→ │ LLM         │
│ (图像编码器) │     │ (MLP)        │     │ (Vicuna等)   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                ↑
                                          ┌─────────────┐
                                          │ 文本 Token   │
                                          └─────────────┘

输入 = [图像 tokens] + [文本 tokens]
输出 = 文本回答

对比 BLIP：
- BLIP: 专用模型，只能做 caption/VQA
- LLaVA: 通用多模态对话，能理解复杂指令、推理、描述细节
"""

import torch
from PIL import Image
import requests

# ============================================================
# 一、LLaVA 基础使用
# ============================================================

def demo_llava_basic():
    print("=" * 60)
    print("一、LLaVA 基础多模态对话")
    print("=" * 60)

    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        # 使用低CPU内存加载模式，通过分片加载和内存映射减少内存占用
        # 适合内存有限的设备，但加载速度会稍慢
        low_cpu_mem_usage=True, # 只在模型加载时生效，不影响推理
    )

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 构造对话
    # LLaVA 使用 chat template 格式
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this image?"},
            ],
        },
    ]

    # 应用 chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(f"生成的 prompt:\n{prompt}\n")

    # 处理输入
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # 生成回答
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

    # 只解码新生成的部分
    output_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    print(f"模型回答: {output_text}")


# ============================================================
# 二、多轮对话
# ============================================================

def demo_multi_turn_conversation():
    print("\n" + "=" * 60)
    print("二、多轮多模态对话")
    print("=" * 60)

    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 多轮对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        },
    ]

    # 第一轮
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    response_1 = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    print(f"用户: Describe this image in detail.")
    print(f"模型: {response_1}\n")

    # 第二轮（追加到对话历史）
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_1}],
    })
    conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": "How many animals are in the image?"}],
    })

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response_2 = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    print(f"用户: How many animals are in the image?")
    print(f"模型: {response_2}")


# ============================================================
# 三、不同任务类型
# ============================================================

def demo_various_tasks():
    print("\n" + "=" * 60)
    print("三、LLaVA 支持的任务类型")
    print("=" * 60)

    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    tasks = [
        ("图像描述", "Describe what you see in this image."),
        ("视觉问答", "What are the cats doing?"),
        ("空间理解", "Where are the cats located in the image?"),
        ("计数", "How many cats can you see?"),
        ("推理", "Why do you think the cats are in this position?"),
        ("OCR/文字", "Is there any text visible in the image?"),
    ]

    for task_name, question in tasks:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        response = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        print(f"[{task_name}]")
        print(f"  Q: {question}")
        print(f"  A: {response}\n")


# ============================================================
# 四、理解 LLaVA 的模型结构
# ============================================================

def demo_model_architecture():
    print("\n" + "=" * 60)
    print("四、LLaVA 模型架构解析")
    print("=" * 60)

    from transformers import LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # 主要组件
    print("LLaVA 主要组件:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params / 1e9:.2f}B 参数")

    # 视觉编码器
    print(f"\n视觉编码器: {type(model.vision_tower.vision_tower).__name__}")
    print(f"  隐藏层维度: {model.config.vision_config.hidden_size}")
    print(f"  图像尺寸: {model.config.vision_config.image_size}")
    print(f"  Patch 尺寸: {model.config.vision_config.patch_size}")

    # 投影层
    print(f"\n投影层 (Multi Modal Projector):")
    print(f"  类型: {model.config.projector_hidden_act}")
    print(f"  结构: Linear(vision_hidden → llm_hidden)")

    # LLM
    print(f"\n语言模型:")
    print(f"  类型: {type(model.language_model).__name__}")
    print(f"  词表大小: {model.config.text_config.vocab_size}")
    print(f"  隐藏层维度: {model.config.text_config.hidden_size}")
    print(f"  层数: {model.config.text_config.num_hidden_layers}")

    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e9:.2f}B")


# ============================================================
# 五、多模态模型对比总结
# ============================================================

def demo_model_comparison():
    print("\n" + "=" * 60)
    print("五、多模态模型对比总结")
    print("=" * 60)

    comparison = """
┌──────────────────────────────────────────────────────────────────────┐
│                    多模态模型对比                                      │
├──────────────┬──────────────┬──────────────────┬─────────────────────┤
│ 模型          │ 架构          │ 能力              │ 适用场景             │
├──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ CLIP         │ 双塔编码器    │ 图文对齐          │ 零样本分类、检索      │
│              │ ViT + Trans  │ 无生成能力         │ 特征提取             │
├──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ BLIP         │ 编码器-解码器  │ Caption + VQA    │ 图像描述、视觉问答    │
│              │ ViT + BERT   │ 专用任务           │                     │
├──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ Stable       │ 扩散模型      │ 文生图            │ 图像生成             │
│ Diffusion    │ U-Net + VAE  │ 图生图、修复       │ 创意设计             │
├──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ LLaVA        │ ViT + LLM    │ 多模态对话        │ 复杂理解、推理        │
│              │ CLIP + Vicuna│ 通用指令遵循       │ 多轮对话             │
├──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ GPT-4V/Qwen-VL │ 大型多模态  │ 全能              │ 通用多模态助手        │
│              │ 闭源/开源     │ 理解+生成+推理     │                     │
└──────────────┴──────────────┴──────────────────┴─────────────────────┘

学习路径：
  CLIP (理解对齐) → BLIP (生成基础) → Diffusion (图像生成) → LLaVA (多模态对话)

推荐资源：
  - HuggingFace 官方教程: https://huggingface.co/docs/transformers/main/en/tasks/image-to-text
  - CLIP 论文: https://arxiv.org/abs/2103.00020
  - BLIP 论文: https://arxiv.org/abs/2201.12086
  - LLaVA 论文: https://arxiv.org/abs/2304.08485
  - Stable Diffusion 论文: https://arxiv.org/abs/2112.10752
  - Diffusers 库文档: https://huggingface.co/docs/diffusers
"""
    print(comparison)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print("注意：LLaVA-1.5-7B 需要约 14GB 内存/显存")
    print("如果内存不足，可以使用 llava-hf/llava-1.5-7b-hf 的量化版本\n")

    # demo_llava_basic()
    # demo_multi_turn_conversation()
    # demo_various_tasks()
    demo_model_architecture()
    # demo_model_comparison()
