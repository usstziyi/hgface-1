"""
多模态学习 3：Stable Diffusion — 文生图
难度：⭐⭐⭐ 中高

Stable Diffusion 是潜在扩散模型 (Latent Diffusion Model)。
核心思想：在潜空间（而非像素空间）进行扩散和去噪。

流程：
1. 文本编码器 (CLIP) 将 prompt 编码为文本嵌入
2. U-Net 在潜空间中去噪（逐步去除噪声）
3. VAE 解码器将潜空间表示解码为图像

关键组件：
- text_encoder: CLIP ViT（文本理解）
- unet: U-Net（去噪网络）
- vae: Variational Autoencoder（图像压缩/解码）
- scheduler: 控制去噪步骤
"""

import torch
from PIL import Image

# ============================================================
# 一、基础文生图
# ============================================================

def demo_basic_text_to_image():
    print("=" * 60)
    print("一、基础文生图 (Text-to-Image)")
    print("=" * 60)

    from diffusers import StableDiffusionPipeline


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 加载 pipeline（首次会下载模型，约 4GB）
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,  # Mac 用 float32，有 CUDA 可用 float16
    )
    # Mac 用户如果内存不足，可以启用 CPU offload
    pipe.enable_attention_slicing()

    # 查看 pipeline 包含的组件
    print("Pipeline 组件:")
    for name in pipe.components:
        print(f"  {name}: {type(pipe.components[name]).__name__}")

    # 生成图像
    prompt = "a photo of a cat sitting on a couch, realistic, high quality"
    negative_prompt = "blurry, low quality, distorted"

    print(f"\nPrompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,   # 去噪步数（越多越精细，但越慢）
        guidance_scale=7.5,       # 文本引导强度（越高越遵循 prompt）
        width=512,
        height=512,
    ).images[0]
    
    image.save("./data/output_sd_basic.png")
    print(f"图像已保存到 ./data/output_sd_basic.png, 尺寸: {image.size}")


# ============================================================
# 二、理解 Pipeline 内部流程
# ============================================================

def demo_pipeline_internals():
    print("\n" + "=" * 60)
    print("二、Pipeline 内部流程解析")
    print("=" * 60)

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )

    # 1. 文本编码
    print("--- Step 1: 文本编码 ---")
    prompt = "a beautiful sunset over the ocean"
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    print(f"Token IDs shape: {text_inputs.input_ids.shape}")  # [1, 77]

    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
    print(f"文本嵌入 shape: {text_embeddings.shape}")  # [1, 77, 768]
    # 77 = token 数, 768 = CLIP 隐藏层维度

    # 2. 初始化潜空间噪声
    print("\n--- Step 2: 初始化潜空间 ---")
    # VAE 的缩放因子（潜空间 vs 像素空间的比例）
    vae_scale_factor = pipe.vae_scale_factor  # 通常是 8
    latent_shape = (1, 4, 512 // vae_scale_factor, 512 // vae_scale_factor)
    print(f"潜空间 shape: {latent_shape}")  # [1, 4, 64, 64]
    # 4 = 潜空间通道数（远小于 RGB 的 3 通道在像素空间的表示）
    # 64x64 = 512/8 x 512/8

    latents = torch.randn(latent_shape)
    print(f"初始噪声 latents shape: {latents.shape}")

    # 3. 去噪过程
    print("\n--- Step 3: 去噪过程 ---")
    pipe.scheduler.set_timesteps(20)
    timesteps = pipe.scheduler.timesteps
    print(f"去噪步数: {len(timesteps)}")
    print(f"前 5 个 timestep: {timesteps[:5].tolist()}")
    print("（timestep 从大到小，噪声逐渐减少）")

    # 4. VAE 解码
    print("\n--- Step 4: VAE 解码 ---")
    print(f"潜空间 [1, 4, 64, 64] → VAE 解码 → 图像 [1, 3, 512, 512]")
    print(f"缩放因子: {vae_scale_factor}x")


# ============================================================
# 三、关键参数调优
# ============================================================

def demo_parameter_tuning():
    print("\n" + "=" * 60)
    print("三、关键参数调优")
    print("=" * 60)

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )

    prompt = "a cat"

    # --- guidance_scale 对比 ---
    print("--- guidance_scale 对比 ---")
    print("guidance_scale 控制图像与文本的匹配程度")
    for scale in [1.0, 5.0, 7.5, 15.0]:
        image = pipe(
            prompt=prompt,
            num_inference_steps=15,
            guidance_scale=scale,
            width=256,
            height=256,
        ).images[0]
        image.save(f"./output_sd_guidance_{scale}.png")
        print(f"  guidance_scale={scale}: 已保存")

    # --- num_inference_steps 对比 ---
    print("\n--- num_inference_steps 对比 ---")
    print("步数越多越精细，但速度越慢")
    for steps in [10, 20, 50]:
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            width=256,
            height=256,
        ).images[0]
        image.save(f"./output_sd_steps_{steps}.png")
        print(f"  steps={steps}: 已保存")

    # --- 随机种子对比 ---
    print("\n--- 随机种子对比 ---")
    print("相同参数，不同种子 → 不同图像")
    for seed in [42, 123, 456]:
        generator = torch.Generator().manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=15,
            guidance_scale=7.5,
            generator=generator,
            width=256,
            height=256,
        ).images[0]
        image.save(f"./output_sd_seed_{seed}.png")
        print(f"  seed={seed}: 已保存")


# ============================================================
# 四、图生图 (Image-to-Image)
# ============================================================

def demo_image_to_image():
    print("\n" + "=" * 60)
    print("四、图生图 (Image-to-Image)")
    print("=" * 60)

    from diffusers import StableDiffusionImg2ImgPipeline
    import requests

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )

    # 加载初始图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    init_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    init_image = init_image.resize((512, 512))

    prompt = "a cat wearing a party hat, cartoon style"

    # strength 控制改变程度：0 = 保持原图, 1 = 完全重绘
    for strength in [0.3, 0.5, 0.8]:
        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]
        image.save(f"./output_sd_img2img_{strength}.png")
        print(f"strength={strength}: 已保存")

    print("\nstrength 参数说明:")
    print("  0.3: 轻微修改，保留原图结构")
    print("  0.5: 中等修改，部分保留原图")
    print("  0.8: 大幅修改，仅保留大致构图")


# ============================================================
# 五、图像局部编辑 (Inpainting)
# ============================================================

def demo_inpainting():
    print("\n" + "=" * 60)
    print("五、图像局部编辑 (Inpainting)")
    print("=" * 60)

    from diffusers import StableDiffusionInpaintPipeline
    import requests

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5-inpainting",
        torch_dtype=torch.float32,
    )

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image.resize((512, 512))

    # 创建 mask（白色区域会被替换）
    # 这里创建一个简单的矩形 mask
    mask = Image.new("L", (512, 512), 0)  # 全黑
    # 在中间画一个白色矩形
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([100, 100, 400, 400], fill=255)

    prompt = "a beautiful flower arrangement"

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    result.save("./output_sd_inpaint.png")
    print("局部编辑完成，已保存到 ./output_sd_inpaint.png")
    print("白色 mask 区域被替换为 'a beautiful flower arrangement'")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print("注意：Stable Diffusion 需要较多内存（建议 8GB+ GPU 或 16GB+ 内存）")
    print("Mac 用户可以取消注释 pipe.enable_attention_slicing() 减少内存占用\n")

    demo_basic_text_to_image()
    # demo_pipeline_internals()
    # demo_parameter_tuning()
    # demo_image_to_image()
    # demo_inpainting()
