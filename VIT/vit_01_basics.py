"""
ViT 学习 1：Vision Transformer 基础原理
难度：⭐ 基础

Vision Transformer (ViT) 将 Transformer 架构直接应用于图像。
核心思想：将图像分割为 patch，将 patch 视为"token"。

与 CNN 的区别：
- CNN: 通过卷积核提取局部特征，逐层扩大感受野
- ViT: 将图像切分为 patch，用 Self-Attention 捕获全局关系

论文: "An Image is Worth 16x16 Words" (2020)
GitHub: https://github.com/google-research/vision_transformer
"""

import torch
import torch.nn as nn
import math

# ============================================================
# 一、Patch Embedding：图像 → Patch 序列
# ============================================================

def demo_patch_embedding():
    print("=" * 60)
    print("一、Patch Embedding：图像 → Patch 序列")
    print("=" * 60)

    # 假设输入图像 （1, 3, 224, 224）
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    
    image = torch.randn(batch_size, channels, height, width)
    print(f"输入图像 shape: {image.shape}")  # [1, 3, 224, 224]

    # ViT 参数
    patch_size = 16  # 每个 patch 的大小
    embed_dim = 768  # 嵌入维度

    # 计算 patch 数量
    num_patches_h = height // patch_size  # 224 / 16 = 14
    num_patches_w = width // patch_size   # 224 / 16 = 14
    num_patches = num_patches_h * num_patches_w  # 14 * 14 = 196

    print(f"Patch 大小: {patch_size}x{patch_size}")
    print(f"Patch 数量: {num_patches_h} x {num_patches_w} = {num_patches}")

    # 方法 1：使用 Conv2d 实现 Patch Embedding（最常用）
    # 等价于将图像切分为 patch 并线性投影
    # 卷积核参数维度：[768, 3, 16, 16]
    # 输出维度：[1, 768, 14, 14]
    patch_embed = nn.Conv2d(
        in_channels=channels,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size,  # stride = kernel_size 确保不重叠
    )
    
    # image (1, 3, 224, 224)
    # patches （1, 768, 14, 14）
    patches = patch_embed(image)
    print(f"\nConv2d 输出 shape: {patches.shape}")  # [1, 768, 14, 14]
    
    # 展平为序列
    # [B, C, H, W] → [B, C, H*W] → [B, H*W, C]
    # PyTorch 默认是 C 语言风格（行优先/Row-Major），
    # 展平顺序是：第 0 行从左到右，再第 1 行从左到右……
    patches = patches.flatten(2).transpose(1, 2)
    print(f"展平后 shape: {patches.shape}")  # [1, 196, 768]
    print(f"  196 = 14 x 14 (patch 数量)")
    print(f"  768 = 嵌入维度")

    # 方法 2：手动实现（帮助理解）
    print("\n--- 手动实现 Patch Embedding ---")
    # 使用 unfold 提取 patch
    # image (1, 3, 224, 224)
    # tensor.unfold(dimension, size, step)
    # unfold 2 (1, 3, 14, 224, 16)
    # unfold 3 (1, 3, 14, 14, 16, 16)
    # patches_manual (1, 3, 14, 14, 16, 16)
    patches_manual = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    print(f"unfold 后 shape: {patches_manual.shape}")  # [1, 3, 14, 14, 16, 16]
    # [B, C, num_h, num_w, patch_h, patch_w]
    
    # 重排并展平
    # [1, 3, 14, 14, 16, 16] → [1, 14, 14, 3, 16, 16] 
    # permute 只是改变维度的"视图"，不移动内存数据。调用 .contiguous() 确保张量在内存中连续存储，这样 .view() 才能正常工作。
    patches_manual = patches_manual.permute(0, 2, 3, 1, 4, 5).contiguous()
    # → [1, 14 * 14, 3 * 16 * 16] = [1, 196, 768]
    patches_manual = patches_manual.view(batch_size, num_patches, -1)
    print(f"展平后 shape: {patches_manual.shape}")  # [1, 196, 768]
    # [B, num_patches, C * patch_h * patch_w]
    
    # 线性投影
    linear_proj = nn.Linear(patch_size * patch_size * channels, embed_dim)
    patches_projected = linear_proj(patches_manual)
    print(f"线性投影后 shape: {patches_projected.shape}")  # [1, 196, 768]


# ============================================================
# 二、CLS Token 与 Position Embedding
# ============================================================

def demo_cls_and_position():
    print("\n" + "=" * 60)
    print("二、CLS Token 与 Position Embedding")
    print("=" * 60)

    batch_size = 1
    num_patches = 196
    embed_dim = 768

    # 模拟 patch embeddings
    patch_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    print(f"Patch embeddings shape: {patch_embeddings.shape}")

    # 1. CLS Token
    # 在序列开头添加一个可学习的 [CLS] token
    # 用于聚合全局信息，类似 BERT
    cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    print(f"\nCLS token shape: {cls_token.shape}")  # [1, 1, 768]

    # 扩展 batch 维度
    cls_tokens = cls_token.expand(batch_size, -1, -1)
    print(f"扩展后 shape: {cls_tokens.shape}")  # [1, 1, 768]

    # 拼接 CLS token 和 patch embeddings
    embeddings_with_cls = torch.cat([cls_tokens, patch_embeddings], dim=1)
    print(f"拼接后 shape: {embeddings_with_cls.shape}")  # [1, 197, 768]
    # 197 = 1 (CLS) + 196 (patches)

    # 2. Position Embedding(1维可学习位置编码)
    # 为每个位置添加可学习的位置编码
    # 序列长度 = 1 (CLS) + num_patches
    pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
    print(f"\nPosition embedding shape: {pos_embedding.shape}")  # [1, 197, 768]

    # 添加位置编码
    final_embeddings = embeddings_with_cls + pos_embedding
    print(f"最终输入 shape: {final_embeddings.shape}")  # [1, 197, 768]

    print("\n关键点:")
    print("  - CLS token: 用于分类任务，聚合全局信息")
    print("  - Position embedding: 提供位置信息（Transformer 本身无位置感知）")
    print("  - 两者都是可学习参数")


# ============================================================
# 三、Transformer Encoder 结构
# ============================================================

def demo_transformer_encoder():
    print("\n" + "=" * 60)
    print("三、Transformer Encoder 结构")
    print("=" * 60)

    embed_dim = 768
    num_heads = 12
    # MLP 隐藏层维度相对于嵌入维度的倍数
    # 例如：embed_dim=768 时，MLP 隐藏层 = 768 * 4 = 3072
    mlp_ratio = 4.0
    dropout = 0.0

    # 1. Multi-Head Self-Attention
    print("--- Multi-Head Self-Attention ---")
    attention = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    
    # 模拟输入
    batch_size = 1
    seq_len = 197  # 1 CLS + 196 patches
    x = torch.randn(batch_size, seq_len, embed_dim) # [1, 197, 768]
    
    # Self-Attention: Q = K = V = x
    # attention 返回两个值：
    # 1. attn_output: Self-Attention 的输出，shape 为 [batch_size, seq_len, embed_dim]
    #    表示每个 token 经过注意力机制后的新表示，融合了全局信息
    # 2. attn_weights: 注意力权重矩阵，shape 为 [batch_size, seq_len, seq_len]
    #    表示每个 token 对其他所有 token 的注意力分数（注意力图）
    attn_output, attn_weights = attention(x, x, x, need_weights=True)
    print(f"输入 shape: {x.shape}")  # [1, 197, 768]
    print(f"Attention 输出 shape: {attn_output.shape}")  # [1, 197, 768]
    print(f"Attention weights shape: {attn_weights.shape}")  # [1, 197, 197]
    # 197x197: 每个 token 对其他所有 token 的注意力权重


    # 2. Feed-Forward Network (FFN)
    print("\n--- Feed-Forward Network ---")
    mlp = nn.Sequential(
        nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        nn.Dropout(dropout),
    )
    
    mlp_output = mlp(attn_output)
    print(f"MLP 隐藏层维度: {int(embed_dim * mlp_ratio)}")  # 3072
    print(f"MLP 输出 shape: {mlp_output.shape}")  # [1, 197, 768]

    # 3. 完整的 Transformer Encoder Block
    print("\n--- 完整 Transformer Encoder Block ---")
    
    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, mlp_ratio, dropout=0.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(dropout),
            )
        
        # pre-normalization,不需要warmup
        def forward(self, x):
            # Pre-norm 结构（ViT 使用）
            # 先 LayerNorm，再 Attention
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            # 先 LayerNorm，再 MLP
            x = x + self.mlp(self.norm2(x))
            return x
    
    block = TransformerBlock(embed_dim, num_heads, mlp_ratio)
    output = block(x)
    print(f"Transformer Block 输出 shape: {output.shape}")  # [1, 197, 768]
    
    print("\n关键设计:")
    print("  - Pre-norm: 先 LayerNorm 再 Attention/MLP（更稳定）")
    print("  - Residual connection: x = x + Attention(x)")
    print("  - GELU 激活函数（比 ReLU 更平滑）")


# ============================================================
# 四、完整的 ViT 模型（简化版）
# ============================================================

class SimpleViT(nn.Module):
    """简化的 Vision Transformer 实现"""
    
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position Embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, embed_dim)
        )
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+num_patches, embed_dim]
        
        # 添加 Position Embedding
        x = x + self.pos_embed
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x) # [B, 1+num_patches, embed_dim]
        
        # 使用 CLS token 的表示进行分类
        cls_output = x[:, 0]  # [B, embed_dim]
        logits = self.head(cls_output)  # [B, num_classes]
        
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


def demo_complete_vit():
    print("\n" + "=" * 60)
    print("四、完整的 ViT 模型（简化版）")
    print("=" * 60)

    # 创建模型
    model = SimpleViT(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M")
    
    # 各模块参数量
    print("\n各模块参数量:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)  # batch_size=2
    logits = model(x)
    print(f"\n输入 shape: {x.shape}")
    print(f"输出 logits shape: {logits.shape}")  # [2, 1000]


# ============================================================
# 五、ViT 的不同配置
# ============================================================

def demo_vit_configurations():
    print("\n" + "=" * 60)
    print("五、ViT 的不同配置")
    print("=" * 60)

    configs = {
        "ViT-Base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "params": "86M",
        },
        "ViT-Large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "params": 307,
        },
        "ViT-Huge": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "params": 632,
        },
    }
    
    print("\nViT 配置对比:")
    print(f"{'模型':15} {'嵌入维度':10} {'层数':6} {'头数':6} {'参数量':10}")
    print("-" * 50)
    for name, cfg in configs.items():
        params = cfg['params']
        if isinstance(params, int):
            params = f"{params}M"
        print(f"{name:15} {cfg['embed_dim']:10} {cfg['depth']:10} "
              f"{cfg['num_heads']:10} {params:10}")
    
    print("\n常用变体:")
    print("  - ViT-B/16: Base, patch_size=16 (最常用)")
    print("  - ViT-B/32: Base, patch_size=32 (更快，精度略低)")
    print("  - ViT-L/16: Large, patch_size=16 (更强，更慢)")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # demo_patch_embedding()
    # demo_cls_and_position()
    # demo_transformer_encoder()
    # demo_complete_vit()
    demo_vit_configurations()
