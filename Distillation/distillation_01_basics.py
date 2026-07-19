"""
模型蒸馏学习 1：蒸馏基础 — 概念、温度缩放、软标签
难度：⭐ 基础

知识蒸馏 (Knowledge Distillation) 的核心思想：
- 将大模型（教师）的知识迁移到小模型（学生）
- 学生模型更小、更快，但性能接近教师

经典论文: "Distilling the Knowledge in a Neural Network" (Hinton, 2015)
GitHub: https://github.com/peterliht/knowledge-distillation-pytorch

核心概念：
1. 软标签 (Soft Labels)：教师模型的输出概率分布
2. 温度缩放 (Temperature Scaling)：控制概率分布的"软度"
3. 蒸馏损失：学生同时学习真实标签和教师的软标签
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 一、温度缩放 (Temperature Scaling)
# ============================================================

def demo_temperature_scaling():
    print("=" * 60)
    print("一、温度缩放 (Temperature Scaling)")
    print("=" * 60)

    # 模拟 logits（模型原始输出）
    logits = torch.tensor([2.0, 1.0, 0.1, -0.5, -1.0])
    
    print(f"原始 logits: {logits.tolist()}")
    
    # 不同温度下的 softmax
    temperatures = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    print("\n不同温度下的概率分布:")
    print(f"{'温度':>6} | {'概率分布':50} | {'熵':.4f}")
    print("-" * 70)
    
    for T in temperatures:
        # 温度缩放：logits / T
        soft_labels = F.softmax(logits / T, dim=-1)
        entropy = -(soft_labels * torch.log(soft_labels + 1e-10)).sum().item()
        
        probs_str = ", ".join([f"{p:.3f}" for p in soft_labels.tolist()])
        print(f"{T:>6.1f} | [{probs_str}] | {entropy:.4f}")
    
    print("\n关键观察:")
    print("  - T=1: 标准 softmax，概率分布较尖锐")
    print("  - T 增大: 分布变平滑，类别间的相对关系更明显")
    print("  - T→∞: 均匀分布")
    print("  - 高温度的软标签包含更多'暗知识' (dark knowledge)")
    print("    即类别间的相似性信息")


# ============================================================
# 二、暗知识 (Dark Knowledge)
# ============================================================

def demo_dark_knowledge():
    print("\n" + "=" * 60)
    print("二、暗知识 (Dark Knowledge)")
    print("=" * 60)

    # 场景：图像分类（猫、狗、汽车、飞机）
    # 真实标签：猫
    logits_teacher = torch.tensor([5.0, 3.5, -1.0, -2.0])  # 猫 > 狗 >> 汽车 > 飞机
    labels = ["猫", "狗", "汽车", "飞机"]
    
    print("场景: 输入图像是一只猫")
    print(f"教师模型 logits: {logits_teacher.tolist()}")
    
    # 硬标签 (one-hot)
    hard_label = torch.tensor([1.0, 0.0, 0.0, 0.0])
    print(f"\n硬标签 (one-hot): {hard_label.tolist()}")
    print("  问题: 只告诉学生'这是猫'，没有更多信息")
    
    # 软标签 (T=5)
    T = 5.0
    soft_labels = F.softmax(logits_teacher / T, dim=-1)
    print(f"\n软标签 (T={T}): {[f'{p:.4f}' for p in soft_labels.tolist()]}")
    print("  包含的信息:")
    print("    - 猫的概率最高 (正确答案)")
    print("    - 狗的概率次高 (猫和狗相似)")
    print("    - 汽车和飞机概率很低 (与猫不相似)")
    
    print("\n这就是'暗知识':")
    print("  - 类别间的相似性关系")
    print("  - 硬标签无法表达这种关系")
    print("  - 学生模型可以从软标签中学到这些知识")


# ============================================================
# 三、蒸馏损失函数
# ============================================================

def demo_distillation_loss():
    print("\n" + "=" * 60)
    print("三、蒸馏损失函数")
    print("=" * 60)

    # 模拟数据
    logits_teacher = torch.tensor([[5.0, 3.5, -1.0, -2.0]])  # 教师输出
    logits_student = torch.tensor([[2.0, 1.5, 0.5, 0.0]])    # 学生输出
    true_labels = torch.tensor([0])  # 真实标签：类别 0
    
    T = 5.0  # 温度
    alpha = 0.7  # 蒸馏损失权重
    beta = 0.3  # 硬标签损失权重
    
    print(f"教师 logits: {logits_teacher[0].tolist()}")
    print(f"学生 logits: {logits_student[0].tolist()}")
    print(f"真实标签: {true_labels.item()}")
    print(f"温度 T: {T}")
    print(f"蒸馏权重 α: {alpha}")
    print(f"硬标签权重 β: {beta}")
    
    # 1. 硬标签损失 (交叉熵)
    loss_hard = F.cross_entropy(logits_student, true_labels)
    print(f"\n1. 硬标签损失 (CE): {loss_hard.item():.4f}")
    
    # 2. 软标签损失 (KL 散度)
    # 教师和学生的软标签
    teacher_soft = F.softmax(logits_teacher / T, dim=-1)
    student_soft = F.softmax(logits_student / T, dim=-1)
    
    # KL 散度
    loss_soft = F.kl_div(
        student_soft.log(),
        teacher_soft,
        reduction='batchmean'
    ) * (T ** 2)  # 乘以 T^2 补偿温度缩放
    
    print(f"2. 软标签损失 (KL): {loss_soft.item():.4f}")
    print(f"   注: 乘以 T^2 使梯度尺度与硬标签损失一致")
    
    # 3. 总蒸馏损失
    loss_total = alpha * loss_soft + beta * loss_hard
    print(f"\n3. 总蒸馏损失: {loss_total.item():.4f}")
    print(f"   = {alpha} × {loss_soft.item():.4f} + {beta} × {loss_hard.item():.4f}")


# ============================================================
# 四、简单的蒸馏实现
# ============================================================

class TeacherModel(nn.Module):
    """教师模型（大模型）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    
    def forward(self, x):
        return self.net(x)


class StudentModel(nn.Module):
    """学生模型（小模型）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.net(x)


def demo_simple_distillation():
    print("\n" + "=" * 60)
    print("四、简单的蒸馏实现")
    print("=" * 60)

    # 创建模型
    teacher = TeacherModel()
    student = StudentModel()
    
    # 统计参数量
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"教师模型参数: {teacher_params:,}")
    print(f"学生模型参数: {student_params:,}")
    print(f"压缩比: {teacher_params / student_params:.1f}x")
    
    # 模拟数据
    batch_size = 32
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, 10, (batch_size,))
    
    # 教师推理（冻结）
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(x)
    
    print(f"\n教师 logits shape: {teacher_logits.shape}")
    
    # 学生训练
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    T = 5.0
    alpha = 0.7
    
    # 前向传播
    student_logits = student(x)
    
    # 计算损失
    loss_hard = F.cross_entropy(student_logits, y)
    loss_soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)
    
    loss = alpha * loss_soft + (1 - alpha) * loss_hard
    
    print(f"\n蒸馏训练:")
    print(f"  硬标签损失: {loss_hard.item():.4f}")
    print(f"  软标签损失: {loss_soft.item():.4f}")
    print(f"  总损失: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("\n蒸馏流程:")
    print("  1. 教师模型冻结，只进行推理")
    print("  2. 学生模型训练，学习教师的软标签")
    print("  3. 同时保留少量硬标签损失")


# ============================================================
# 五、蒸馏的变体
# ============================================================

def demo_distillation_variants():
    print("\n" + "=" * 60)
    print("五、蒸馏的变体")
    print("=" * 60)

    variants = """
    知识蒸馏的主要类型：
    
    ┌─────────────────┬──────────────────────────────────────────┐
    │ 类型             │ 描述                                      │
    ├─────────────────┼──────────────────────────────────────────┤
    │ Logit 蒸馏      │ 学生拟合教师的输出 logits/概率分布          │
    │ (Hinton 2015)   │ 最经典的蒸馏方法                           │
    ├─────────────────┼──────────────────────────────────────────┤
    │ 特征蒸馏        │ 学生拟合教师的中间层特征表示                │
    │ (Hint 2015)     │ 也叫 Hint-based distillation             │
    ├─────────────────┼──────────────────────────────────────────┤
    │ 关系蒸馏        │ 学生拟合教师样本间的关系                    │
    │ (PKT 2018)      │ 如样本间的相似度矩阵                       │
    ├─────────────────┼──────────────────────────────────────────┤
    │ 自蒸馏          │ 模型自己蒸馏自己                           │
    │ (Born Again)    │ 无需额外的教师模型                         │
    ├─────────────────┼──────────────────────────────────────────┤
    │ 在线蒸馏        │ 教师和学生同时训练                         │
    │ (Deep Mutual)   │ 互相学习，共同进步                         │
    └─────────────────┴──────────────────────────────────────────┘
    
    应用场景：
    - 模型压缩：大模型 → 小模型，部署到边缘设备
    - 加速推理：减少计算量，保持性能
    - 多任务学习：教师集成多个任务的知识
    - 数据高效学习：利用教师的暗知识
    """
    print(variants)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_temperature_scaling()
    demo_dark_knowledge()
    demo_distillation_loss()
    demo_simple_distillation()
    demo_distillation_variants()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    知识蒸馏的核心要点：
    
    1. 温度缩放：控制概率分布的"软度"
       - 高温度 → 更平滑的分布 → 更多暗知识
    
    2. 蒸馏损失 = α × 软标签损失 + β × 硬标签损失
       - 软标签：教师的输出分布
       - 硬标签：真实标签
    
    3. 蒸馏的优势：
       - 学生模型更小、更快
       - 性能接近教师
       - 可以部署到资源受限的设备
    
    4. 关键超参数：
       - 温度 T：通常 3-20
       - α：蒸馏权重，通常 0.5-0.9
    """)
