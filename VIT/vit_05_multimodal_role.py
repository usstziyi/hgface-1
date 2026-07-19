"""
ViT 学习 5：ViT 在多模态模型中的角色
难度：⭐⭐⭐⭐⭐ 高级

ViT 是大多数多模态模型的视觉 backbone。
本文件探讨 ViT 在 CLIP、BLIP、LLaVA 等模型中的具体作用。

核心观点：
- ViT 将图像转换为"视觉 token"
- 这些 token 与文本 token 一起输入到语言模型
- ViT 的质量直接影响多模态模型的性能
"""

import torch
from transformers import (
    CLIPModel,
    BlipForConditionalGeneration,
    LlavaForConditionalGeneration,
)
from PIL import Image
import requests

# ============================================================
# 一、CLIP 中的 ViT
# ============================================================

def demo_clip_vit():
    print("=" * 60)
    print("一、CLIP 中的 ViT")
    print("=" * 60)

    """
    CLIP 架构：
    ┌─────────────┐         ┌──────────────┐
    │ ViT         │ ──────→ │ 投影层        │ ──→ 图像嵌入 [512]
    │ (图像编码器) │         │ (Linear)     │
    └─────────────┘         └──────────────┘
                                    ↓ 余弦相似度
    ┌─────────────┐         ┌──────────────┐
    │ Transformer │ ──────→ │ 投影层        │ ──→ 文本嵌入 [512]
    │ (文本编码器) │         │ (Linear)     │
    └─────────────┘         └──────────────┘
    """

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    
    print(f"模型: {model_name}")
    print(f"\n视觉编码器 (ViT):")
    print(f"  类型: {type(model.vision_model).__name__}")
    print(f"  隐藏层维度: {model.config.vision_config.hidden_size}")
    print(f"  层数: {model.config.vision_config.num_hidden_layers}")
    print(f"  注意力头数: {model.config.vision_config.num_attention_heads}")
    print(f"  Patch 尺寸: {model.config.vision_config.patch_size}")
    print(f"  图像尺寸: {model.config.vision_config.image_size}")
    
    # 计算视觉编码器参数量
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    print(f"  参数量: {vision_params / 1e6:.1f}M")
    
    print(f"\n文本编码器 (Transformer):")
    print(f"  类型: {type(model.text_model).__name__}")
    print(f"  隐藏层维度: {model.config.text_config.hidden_size}")
    print(f"  层数: {model.config.text_config.num_hidden_layers}")
    
    # 投影层
    print(f"\n投影层:")
    print(f"  视觉投影: {model.visual_projection}")
    print(f"  文本投影: {model.text_projection}")

    
    # 测试图像编码
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(model_name)
    inputs = processor(images=image, return_tensors="pt")
    
    # 获取 ViT 的输出
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
    
    print(f"\nViT 输出:")
    print(f"  last_hidden_state: {vision_outputs.last_hidden_state.shape}")
    # [1, 50, 768] = [batch, 1+49 patches, hidden_dim]
    print(f"  pooler_output: {vision_outputs.pooler_output.shape}")
    # [1, 768] = CLS token 经过投影
    
    print("\nCLIP 中 ViT 的作用:")
    print("  - 将 224x224 图像编码为 50 个 token（1 CLS + 49 patches）")
    print("  - CLS token 经过投影得到 512 维图像嵌入")
    print("  - 与文本嵌入计算相似度实现图文匹配")


# ============================================================
# 二、BLIP 中的 ViT
# BLIP（Bootstrapping Language-Image Pre-training）是由 Salesforce Research 团队开发的多模态视觉-语言预训练模型。
# blip-image-captioning-base 是其在 图像描述生成（Image Captioning） 任务上的基础版本，能够自动为图像生成准确、自然的文字描述（即"看图说话"）。
# ============================================================

def demo_blip_vit():
    print("\n" + "=" * 60)
    print("二、BLIP 中的 ViT")
    print("=" * 60)

    """
    BLIP 架构：
    ┌─────────────┐
    │ ViT         │ ──→ 视觉特征
    │ (图像编码器) │         ↓
    └─────────────┘    ┌──────────────┐
                       │ Q-Former     │ ──→ 压缩的视觉 token
                       │ (可选)        │
                       └──────────────┘
                              ↓
    ┌─────────────┐    ┌──────────────┐
    │ Transformer │ ──→│ 文本解码器    │ ──→ 生成的文本
    │ (文本编码器) │    │ (基于 BERT)  │
    └─────────────┘    └──────────────┘
    """

    model_name = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    print(f"模型: {model_name}")
    print(f"\n视觉编码器:")
    print(f"  类型: {type(model.vision_model).__name__}")
    
    # BLIP 的视觉编码器配置
    vision_config = model.config.vision_config
    print(f"  隐藏层维度: {vision_config.hidden_size}")
    print(f"  层数: {vision_config.num_hidden_layers}")
    print(f"  注意力头数: {vision_config.num_attention_heads}")
    print(f"  Patch 尺寸: {vision_config.patch_size}")
    print(f"  图像尺寸: {vision_config.image_size}")
    
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    print(f"  参数量: {vision_params / 1e6:.1f}M")
    
    # 测试图像编码
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    
    from transformers import BlipProcessor
    processor = BlipProcessor.from_pretrained(model_name)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
    
    print(f"\nViT 输出:")
    print(f"  last_hidden_state: {vision_outputs.last_hidden_state.shape}")
    # [1, 577, 768] = [batch, 1+576 patches, hidden_dim]
    # BLIP 使用 384x384 图像，patch_size=16，所以 24x24=576 patches
    
    print("\nBLIP 中 ViT 的作用:")
    print("  - 将图像编码为 patch token 序列")
    print("  - 这些 token 输入到 Q-Former 进行压缩")
    print("  - 压缩后的 token 与文本一起输入解码器")
    print("  - 生成图像描述或回答视觉问题")

def demo_blip_vit_run():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import requests

    # 加载模型和处理器
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # 加载图像
    image_path = "VIT/data/dog.png"
    image = Image.open(image_path).convert("RGB")

    # 无条件生成描述
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print("************************************************")
    print(caption)  # 例如: "two cats sleeping on a couch"

    # 条件生成（给定提示）
    text = "a photography of"
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)  # 例如: "a photography of two cats on a couch"
    print("************************************************")
# ============================================================
# 三、LLaVA 中的 ViT
# ============================================================

def demo_llava_vit():
    print("\n" + "=" * 60)
    print("三、LLaVA 中的 ViT")
    print("=" * 60)

    """
    LLaVA 架构：
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │ CLIP ViT    │ ──→│ 投影层        │ ──→│ LLM         │
    │ (图像编码器) │    │ (MLP)        │    │ (Vicuna)    │
    └─────────────┘    └──────────────┘    └─────────────┘
                              ↑                    ↑
                         视觉 token           文本 token
    
    输入 = [视觉 tokens] + [文本 tokens]
    输出 = 文本回答
    """

    model_name = "llava-hf/llava-1.5-7b-hf"
    
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        print(f"模型: {model_name}")
        
        # 视觉编码器（使用 CLIP ViT）
        print(f"\n视觉编码器 (CLIP ViT):")
        vision_tower = model.vision_tower
        print(f"  类型: {type(vision_tower).__name__}")
        print(f"  内部模型: {type(vision_tower.vision_tower).__name__}")
        
        vision_config = model.config.vision_config
        print(f"  隐藏层维度: {vision_config.hidden_size}")
        print(f"  图像尺寸: {vision_config.image_size}")
        print(f"  Patch 尺寸: {vision_config.patch_size}")
        
        vision_params = sum(p.numel() for p in vision_tower.parameters())
        print(f"  参数量: {vision_params / 1e6:.1f}M")
        
        # 投影层
        print(f"\n投影层 (Multi Modal Projector):")
        projector = model.multi_modal_projector
        print(f"  类型: {type(projector).__name__}")
        projector_params = sum(p.numel() for p in projector.parameters())
        print(f"  参数量: {projector_params / 1e6:.1f}M")
        
        # 语言模型
        print(f"\n语言模型:")
        llm = model.language_model
        print(f"  类型: {type(llm).__name__}")
        llm_params = sum(p.numel() for p in llm.parameters())
        print(f"  参数量: {llm_params / 1e9:.2f}B")
        
        # 总参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n总参数量: {total_params / 1e9:.2f}B")
        
        # 参数占比
        print(f"\n参数占比:")
        print(f"  视觉编码器: {vision_params / total_params:.2%}")
        print(f"  投影层: {projector_params / total_params:.4%}")
        print(f"  语言模型: {llm_params / total_params:.2%}")
        
    except Exception as e:
        print(f"LLaVA 模型加载失败: {e}")
        print("需要约 14GB 内存")
    
    print("\nLLaVA 中 ViT 的作用:")
    print("  - 使用 CLIP ViT 编码图像")
    print("  - 输出 576 个 patch token (24x24)")
    print("  - 经过投影层映射到 LLM 的嵌入空间")
    print("  - 与文本 token 拼接输入 LLM")
    print("  - LLM 基于视觉+文本信息生成回答")

# ============================================================
# 四、ViT 在多模态中的关键作用
# ============================================================

def demo_vit_role_summary():
    print("\n" + "=" * 60)
    print("四、ViT 在多模态中的关键作用")
    print("=" * 60)

    summary = """
    ViT 作为视觉 backbone 的核心价值：
    
    1. 将图像转换为 token 序列
       ┌─────────┐     ┌─────────────────────────────────┐
       │ 图像    │ ──→ │ [CLS] [P1] [P2] ... [P196]      │
       │ 224x224 │     │  768   768  768       768       │
       └─────────┘     └─────────────────────────────────┘
       
    2. 与 NLP 无缝衔接
       - 视觉 token 和文本 token 格式相同
       - 可以直接输入到 Transformer
       - 实现真正的"视觉语言"模型
    
    3. 预训练知识迁移
       - 在大规模图像数据上预训练
       - 学习通用的视觉特征
       - 多模态训练时可以冻结或微调
    
    4. 不同任务选择不同的 ViT
       ┌─────────────────┬──────────────────┬────────────────┐
       │ 多模态模型       │ 视觉 backbone    │ 原因            │
       ├─────────────────┼──────────────────┼────────────────┤
       │ CLIP            │ ViT              │ 需要与文本对齐  │
       │ BLIP            │ ViT              │ 需要强视觉特征  │
       │ LLaVA           │ CLIP ViT         │ 继承 CLIP 能力  │
       │ GPT-4V          │ 未公开（可能是    │ 需要高质量视觉  │
       │                 │ 定制 ViT）       │ 理解能力        │
       └─────────────────┴──────────────────┴────────────────┘
    """
    print(summary)


# ============================================================
# 五、实战：用不同 ViT 做图像描述
# ============================================================

def demo_practical_comparison():
    print("\n" + "=" * 60)
    print("五、实战：不同视觉 backbone 的效果对比")
    print("=" * 60)

    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    # 使用 BLIP 测试不同视觉编码器的效果
    # 注意：实际中需要不同视觉 backbone 的 BLIP 模型
    # 这里演示概念
    
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"图像: {url}")
    print(f"生成的描述: {caption}")
    
    print("\n视觉 backbone 对描述质量的影响:")
    print("  - 更强的 ViT → 更好的视觉理解 → 更准确的描述")
    print("  - ViT-B/16 vs ViT-L/16: 大模型通常更好")
    print("  - 预训练数据量也很重要")
    print("  - CLIP 预训练的 ViT 可能有更好的语义理解")




# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # demo_clip_vit()
    # demo_blip_vit()
    # demo_blip_vit_run()
    # demo_llava_vit()
    # demo_vit_role_summary()
    demo_practical_comparison()
    
    # print("\n" + "=" * 60)
    # print("总结")
    # print("=" * 60)
    # print("""
    # ViT 在多模态中的地位：
    
    # 1. ViT 是多模态模型的"眼睛"
    #    - 负责将图像转换为模型可理解的 token
    
    # 2. ViT 的质量决定多模态模型的上限
    #    - 视觉特征越好，多模态理解越强
    
    # 3. 选择合适的 ViT 很重要
    #    - CLIP ViT: 适合需要图文对齐的任务
    #    - DINOv2: 适合纯视觉特征提取
    #    - Swin: 适合需要多尺度特征的任务
    
    # 4. 未来趋势
    #    - 更大的 ViT（ViT-G, ViT-e）
    #    - 更高效的 ViT（Swin, FastViT）
    #    - 更好的预训练方法（MAE, DINOv2）
    # """)
