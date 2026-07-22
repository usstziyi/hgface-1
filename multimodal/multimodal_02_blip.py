"""
多模态学习 2：BLIP — 图像描述与视觉问答
难度：⭐⭐ 中等

BLIP (Bootstrapping Language-Image Pre-training) 是 Salesforc 提出的
视觉语言模型，支持图像描述生成 (Captioning) 和视觉问答 (VQA)。

架构：
- 图像编码器：ViT
- 文本编码器/解码器：基于 BERT/GPT 的 Transformer
- 三种损失：ITC（图文对比）、ITM（图文匹配）、LM（语言建模）
"""

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
import os
from PIL import Image
import requests

# ============================================================
# 一、图像描述生成 (Image Captioning)
# ============================================================

def demo_image_captioning():
    print("=" * 60)
    print("一、图像描述生成 (Image Captioning)")
    print("=" * 60)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    image_file = os.path.join(data_dir, "cat.jpg")
    image = Image.open(image_file)

    # --- 无条件描述生成 ---
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"无条件描述: {caption}")


    # --- 有条件描述生成 ---
    # 提供文本前缀，引导模型生成特定风格的描述
    text_prompt = "a photography of"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"条件描述 (前缀='{text_prompt}'): {caption}")


    # 尝试不同前缀
    for prompt in ["an image of", "a picture showing", "this image contains"]:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"条件描述 (前缀='{prompt}'): {caption}")
    

    # --- 生成参数调优 ---
    print("\n--- 生成参数对比 ---")
    inputs = processor(images=image, return_tensors="pt")

    # 贪心搜索
    ids_greedy = model.generate(**inputs, max_length=50, num_beams=1)
    print(f"贪心搜索: {processor.batch_decode(ids_greedy, skip_special_tokens=True)[0]}")

    # Beam search（默认）
    ids_beam = model.generate(**inputs, max_length=50, num_beams=4)
    print(f"Beam search: {processor.batch_decode(ids_beam, skip_special_tokens=True)[0]}")

    # 采样（更多样化）
    ids_sample = model.generate(
        **inputs,                    # 输入张量（包含图像特征）
        max_length=50,               # 生成文本的最大长度
        do_sample=True,              # 启用采样模式（而非贪心搜索）
        top_k=50,                    # 只从概率最高的前50个token中采样
        temperature=0.7              # 温度参数，控制生成文本的随机性（越低越保守）
    )
    print(f"采样生成: {processor.batch_decode(ids_sample, skip_special_tokens=True)[0]}")


# ============================================================
# 二、视觉问答 (Visual Question Answering)
# ============================================================

def demo_vqa():
    print("\n" + "=" * 60)
    print("二、视觉问答 (Visual Question Answering)")
    print("=" * 60)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    image_file = os.path.join(data_dir, "cat.jpg")
    image = Image.open(image_file)

    # 不同类型的问题
    questions = [
        "how many cats are in the picture?",
        "what color are the cats?",
        "what are the cats doing?",
        "is there a blanket in the image?",
        "what is in the background?",
    ]

    # 批量处理版本
    images = [image] * len(questions)  # 为每个问题重复图像
    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True)
    generated_ids = model.generate(**inputs, max_length=20)
    answers = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for question, answer in zip(questions, answers):
        print(f"Q: {question}")
        print(f"A: {answer}\n")


# ============================================================
# 三、BLIP-large 模型（更强的能力）
# ============================================================

def demo_blip_large():
    print("\n" + "=" * 60)
    print("三、BLIP-large 模型对比")
    print("=" * 60)

    # base vs large 模型对比
    for model_name in [
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large",
    ]:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        image_file = os.path.join(data_dir, "light.png")
        image = Image.open(image_file)

        inputs = processor(images=image, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 统计参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name.split('/')[-1]}: {num_params / 1e6:.1f}M 参数")
        print(f"  描述: {caption}\n")


# ============================================================
# 四、批量处理多张图像
# ============================================================

def demo_batch_processing():
    print("\n" + "=" * 60)
    print("四、批量图像描述生成")
    print("=" * 60)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    image_files = [
        os.path.join(data_dir, "cat.jpg"),
        os.path.join(data_dir, "dog.png"),
        os.path.join(data_dir, "light.png"),
    ]
    images = [Image.open(url) for url in image_files]

    # 批量处理版本
    inputs = processor(images=images, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=50)
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    for i, caption in enumerate(captions):
        print(f"图像 {i+1}: {caption}")


# ============================================================
# 五、理解 BLIP 的模型结构
# ============================================================

def demo_model_structure():
    print("\n" + "=" * 60)
    print("五、BLIP 模型结构解析")
    print("=" * 60)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # 查看主要模块
    print("模型主要模块:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params / 1e6:.1f}M 参数")

    # 查看 config
    config = model.config
    print(f"\n关键配置:")
    print(f"  图像尺寸: {config.image_size}")
    print(f"  patch 尺寸: {config.patch_size}")
    print(f"  隐藏层维度: {config.hidden_size}")
    print(f"  注意力头数: {config.num_attention_heads}")
    print(f"  编码器层数: {config.num_hidden_layers}")
    print(f"  词表大小: {config.vocab_size}")

    # 了解 vision encoder
    print(f"\n视觉编码器:")
    vision_config = config.vision_config
    print(f"  模型类型: {vision_config.model_type}")
    print(f"  隐藏层维度: {vision_config.hidden_size}")
    print(f"  中间层维度: {vision_config.intermediate_size}")
    print(f"  层数: {vision_config.num_hidden_layers}")
    print(f"  注意力头数: {vision_config.num_attention_heads}")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # demo_image_captioning()
    # demo_vqa()
    # demo_blip_large()
    demo_batch_processing()
    # demo_model_structure()
