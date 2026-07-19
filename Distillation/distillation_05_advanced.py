"""
模型蒸馏学习 5：进阶蒸馏技术
难度：⭐⭐⭐⭐⭐ 高级

涵盖进阶蒸馏技术：
1. 自蒸馏 (Self-Distillation)
2. PKD (Patient Knowledge Distillation)
3. 实际部署技巧
4. 蒸馏在大模型时代的应用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 一、自蒸馏 (Self-Distillation)
# ============================================================

def demo_self_distillation():
    print("=" * 60)
    print("一、自蒸馏 (Self-Distillation)")
    print("=" * 60)

    concept = """
    自蒸馏：模型自己蒸馏自己
    
    传统蒸馏：教师 → 学生（两个独立模型）
    自蒸馏：模型深层 → 模型浅层（同一个模型）
    
    ┌─────────────────────────────────────┐
    │              模型                    │
    │                                     │
    │  输入 → L1 → L2 → L3 → L4 → 输出   │
    │           ↓              ↓          │
    │         学生            教师         │
    │         (浅层)          (深层)       │
    │                                     │
    │  损失 = CE(学生输出, 标签)           │
    │       + KL(学生输出, 教师输出)       │
    └─────────────────────────────────────┘
    
    优势：
    - 不需要额外的教师模型
    - 节省内存和计算
    - 深层作为教师指导浅层
    
    论文: "Born Again Networks" (Furlanello et al., 2018)
    """
    print(concept)

    # 模拟自蒸馏
    class SelfDistillModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(784, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 10),
            ])
        
        def forward(self, x, return_intermediate=False):
            # 浅层输出（学生）
            h1 = F.relu(self.layers[0](x))
            h2 = F.relu(self.layers[1](h1))
            student_logits = self.layers[2](h2)
            
            # 深层输出（教师）
            h3 = F.relu(self.layers[2](h2))
            teacher_logits = self.layers[3](h3)
            
            if return_intermediate:
                return student_logits, teacher_logits
            return teacher_logits
    
    model = SelfDistillModel()
    x = torch.randn(4, 784)
    
    student_logits, teacher_logits = model(x, return_intermediate=True)
    
    print(f"学生 logits shape: {student_logits.shape}")
    print(f"教师 logits shape: {teacher_logits.shape}")
    
    # 自蒸馏损失
    T = 3.0
    loss_hard = F.cross_entropy(teacher_logits, torch.randint(0, 10, (4,)))
    loss_soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)
    
    loss = 0.5 * loss_hard + 0.5 * loss_soft
    print(f"\n自蒸馏损失: {loss.item():.4f}")


# ============================================================
# 二、PKD (Patient Knowledge Distillation)
# ============================================================

def demo_pkd():
    print("\n" + "=" * 60)
    print("二、PKD (Patient Knowledge Distillation)")
    print("=" * 60)

    concept = """
    PKD: 针对 BERT 的蒸馏方法
    
    核心思想：
    - 学生继承教师的每一层
    - 学生层数更少，但每层都学习教师对应层
    
    教师 BERT (12 层):
    L1 → L2 → L3 → ... → L12
    
    学生 BERT (6 层):
    l1 → l2 → l3 → l4 → l5 → l6
     ↓    ↓    ↓    ↓    ↓    ↓
    L2   L4   L6   L8   L10  L12
    
    学生第 i 层学习教师第 2i 层
    
    论文: "Patient Knowledge Distillation for BERT Model Compression"
    """
    print(concept)

    # 模拟 PKD
    class PKDLoss(nn.Module):
        def __init__(self, teacher_layer_indices):
            super().__init__()
            self.teacher_layer_indices = teacher_layer_indices
        
        def forward(self, student_hidden_states, teacher_hidden_states):
            """
            Args:
                student_hidden_states: 学生各层输出列表
                teacher_hidden_states: 教师各层输出列表
            """
            total_loss = 0
            
            for i, teacher_idx in enumerate(self.teacher_layer_indices):
                s_hidden = student_hidden_states[i]
                t_hidden = teacher_hidden_states[teacher_idx]
                
                # MSE 损失
                loss = F.mse_loss(s_hidden, t_hidden.detach())
                total_loss += loss
            
            return total_loss / len(self.teacher_layer_indices)
    
    # 模拟数据
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    
    # 学生 6 层
    student_hidden = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(6)]
    
    # 教师 12 层
    teacher_hidden = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(12)]
    
    # 学生第 i 层对应教师第 2i 层
    teacher_indices = [1, 3, 5, 7, 9, 11]
    
    criterion = PKDLoss(teacher_indices)
    loss = criterion(student_hidden, teacher_hidden)
    
    print(f"\nPKD 损失: {loss.item():.4f}")
    print(f"学生层数: {len(student_hidden)}")
    print(f"教师层数: {len(teacher_hidden)}")
    print(f"层对应关系: 学生[i] → 教师[{teacher_indices[i]}]")


# ============================================================
# 三、蒸馏的实际部署技巧
# ============================================================

def demo_deployment_tips():
    print("\n" + "=" * 60)
    print("三、蒸馏的实际部署技巧")
    print("=" * 60)

    tips = """
    蒸馏部署技巧：
    
    1. 教师模型选择
       - 选择性能好的大模型作为教师
       - 教师不需要是最快的，但要足够准
       - 可以使用模型集成作为教师
    
    2. 学生模型设计
       - 根据部署平台选择架构
       - 移动端：MobileNet, EfficientNet-Lite
       - 服务器端：小型 Transformer
    
    3. 温度选择
       - 通常 T = 3-10
       - 分类任务：T 可以大一些（5-20）
       - 检测任务：T 小一些（1-5）
    
    4. 蒸馏权重
       - α = 0.5-0.9
       - 教师质量高：α 大一些
       - 数据充足：α 小一些
    
    5. 训练策略
       - 先正常训练学生，再蒸馏
       - 或直接蒸馏
       - 可以多次蒸馏（学生再蒸馏学生）
    
    6. 评估指标
       - 准确率
       - 推理速度（FLOPs, 延迟）
       - 模型大小
       - 内存占用
    """
    print(tips)


# ============================================================
# 四、蒸馏在大模型时代的应用
# ============================================================

def demo_llm_distillation():
    print("\n" + "=" * 60)
    print("四、蒸馏在大模型时代的应用")
    print("=" * 60)

    llm_distill = """
    大模型蒸馏：
    
    场景：将 GPT-4、LLaMA-70B 等大模型蒸馏为小模型
    
    方法：
    1. 输出蒸馏
       - 学生拟合教师的输出分布
       - 适用于分类任务
    
    2. 思维链蒸馏 (CoT Distillation)
       - 学生模仿教师的推理过程
       - 适用于复杂推理任务
    
    3. 数据蒸馏
       - 用大模型生成高质量数据
       - 用小模型在生成数据上训练
    
    实际案例：
    - DistilGPT2: GPT-2 的蒸馏版本
    - TinyLlama: LLaMA 风格的小模型
    - Phi-1/2/3: 微软的数据蒸馏模型
    
    数据蒸馏流程：
    ┌─────────────┐
    │ 大模型      │
    │ (GPT-4)     │
    └──────┬──────┘
           │ 生成数据
           ↓
    ┌─────────────┐
    │ 高质量数据  │
    │ (指令+回答) │
    └──────┬──────┘
           │ 训练
           ↓
    ┌─────────────┐
    │ 小模型      │
    │ (Phi-3)     │
    └─────────────┘
    """
    print(llm_distill)


# ============================================================
# 五、蒸馏方法总结
# ============================================================

def demo_summary():
    print("\n" + "=" * 60)
    print("五、蒸馏方法总结")
    print("=" * 60)

    summary = """
    知识蒸馏方法全景：
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                      知识蒸馏                                    │
    ├─────────────┬─────────────┬─────────────┬───────────────────────┤
    │  Logit 蒸馏  │  特征蒸馏    │  关系蒸馏    │  其他                  │
    ├─────────────┼─────────────┼─────────────┼───────────────────────┤
    │ Hinton 2015 │ FitNets     │ PKT         │ 自蒸馏                │
    │ 软标签      │ 中间层对齐   │ 样本关系    │ Born Again            │
    │ 温度缩放    │ Attention   │ 相似度矩阵  │ 在线蒸馏              │
    │             │ Transfer    │             │ Deep Mutual Learning  │
    │             │             │             │                       │
    │             │             │             │ 大模型蒸馏            │
    │             │             │             │ CoT 蒸馏              │
    │             │             │             │ 数据蒸馏              │
    └─────────────┴─────────────┴─────────────┴───────────────────────┘
    
    选择建议：
    
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ 场景             │ 推荐方法                                      │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ 简单分类任务     │ Logit 蒸馏（Hinton）                         │
    │ CNN 压缩         │ 特征蒸馏（Attention Transfer）               │
    │ BERT 压缩        │ PKD + Logit 蒸馏                             │
    │ ViT 压缩         │ DeiT 风格蒸馏                                 │
    │ 无额外教师       │ 自蒸馏                                        │
    │ 大模型压缩       │ 数据蒸馏 + 指令微调                           │
    └─────────────────┴──────────────────────────────────────────────┘
    
    参考资源：
    - 综述: "Knowledge Distillation: A Survey" (2020)
    - GitHub: https://github.com/dkozlov/awesome-knowledge-distillation
    - HuggingFace: 搜索 "distilled" 或 "distillation"
    """
    print(summary)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_self_distillation()
    demo_pkd()
    demo_deployment_tips()
    demo_llm_distillation()
    demo_summary()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    进阶蒸馏技术的核心要点：
    
    1. 自蒸馏：模型自己蒸馏自己，无需额外教师
    
    2. PKD：针对 BERT 的层对层蒸馏
    
    3. 部署技巧：
       - 合理选择温度和权重
       - 根据部署平台设计学生模型
       - 评估准确率 + 速度 + 大小
    
    4. 大模型时代：
       - 数据蒸馏成为主流
       - 用大模型生成数据训练小模型
       - Phi、TinyLlama 等成功案例
    
    5. 未来趋势：
       - 蒸馏 + 量化 + 剪枝 组合使用
       - 自动化蒸馏流程
       - 多模态蒸馏
    """)
