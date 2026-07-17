"""
项目 6：TensorDataset 与 TensorDataLoader 用法
TensorDataset 是最简单的 Dataset 实现，直接包装已有的 tensor。
适用于数据已经预处理为 tensor 的场景（如表格数据、数值特征等）。
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

# ============================================================
# 一、基础用法：用 tensor 创建 Dataset
# ============================================================

def demo_basic():
    print("=" * 60)
    print("一、TensorDataset 基础用法")
    print("=" * 60)

    # 模拟数据：100 个样本，每个样本 10 个特征
    X = torch.randn(100, 10)   # 特征矩阵 [100, 10]
    y = torch.randint(0, 2, (100,))  # 标签 [100]

    # 创建 TensorDataset
    dataset = TensorDataset(X, y)
    print(f"数据集大小: {len(dataset)}")

    # 取单条样本（返回一个 tuple）
    sample_x, sample_y = dataset[0]
    print(f"单条样本 X shape: {sample_x.shape}")  # [10]
    print(f"单条样本 y: {sample_y}")

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"\nDataLoader 批次数: {len(dataloader)}")  # ceil(100/16) = 7

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"\n--- 第一个 Batch ---")
            print(f"  X shape: {batch_x.shape}")  # [16, 10]
            print(f"  y shape: {batch_y.shape}")  # [16]


# ============================================================
# 二、配合模型训练：完整的训练循环
# ============================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def demo_training():
    print("\n" + "=" * 60)
    print("二、配合模型训练（完整训练循环）")
    print("=" * 60)

    # 1. 准备数据
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    # 生成线性可分的数据
    y = (X[:, 0] + X[:, 1] > 0).long()

    dataset = TensorDataset(X, y)

    # 2. 切分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 3. 创建模型
    model = SimpleClassifier(input_dim=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. 训练
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = correct / total
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Acc: {train_acc:.4f}")

    # 5. 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"\n测试集准确率: {correct / total:.4f}")


# ============================================================
# 三、多输入场景：文本 + 数值特征
# ============================================================

def demo_multi_input():
    print("\n" + "=" * 60)
    print("三、多输入 TensorDataset（文本 token ids + 数值特征）")
    print("=" * 60)

    # 模拟场景：文本 token ids + 额外数值特征（如用户年龄、评分等）
    num_samples = 200
    seq_len = 20

    # 文本 token ids [200, 20]
    input_ids = torch.randint(0, 30000, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)

    # 额外数值特征 [200, 3]（如：用户年龄归一化、历史评分、类别编码）
    numeric_features = torch.randn(num_samples, 3)

    # 标签 [200]
    labels = torch.randint(0, 2, (num_samples,))

    # TensorDataset 可以接受任意数量的 tensor
    dataset = TensorDataset(input_ids, attention_mask, numeric_features, labels)
    print(f"数据集大小: {len(dataset)}")

    # 单条样本返回 4 个 tensor
    sample = dataset[0]
    print(f"单条样本包含 {len(sample)} 个 tensor:")
    print(f"  input_ids: {sample[0].shape}")          # [20]
    print(f"  attention_mask: {sample[1].shape}")     # [20]
    print(f"  numeric_features: {sample[2].shape}")   # [3]
    print(f"  label: {sample[3]}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_input_ids, batch_mask, batch_numeric, batch_labels in dataloader:
        print(f"\nBatch shapes:")
        print(f"  input_ids: {batch_input_ids.shape}")           # [32, 20]
        print(f"  attention_mask: {batch_mask.shape}")           # [32, 20]
        print(f"  numeric_features: {batch_numeric.shape}")      # [32, 3]
        print(f"  labels: {batch_labels.shape}")                 # [32]
        break


# ============================================================
# 四、与自定义 Dataset 的对比
# ============================================================

def demo_comparison():
    print("\n" + "=" * 60)
    print("四、TensorDataset vs 自定义 Dataset 对比")
    print("=" * 60)

    X = torch.randn(50, 5)
    y = torch.randint(0, 2, (50,))

    # --- 方式 1：TensorDataset（简单直接） ---
    tensor_ds = TensorDataset(X, y)
    print("TensorDataset:")
    print(f"  代码量: 1 行")
    print(f"  适用场景: 数据已经是 tensor，无需额外处理")
    print(f"  限制: 不能在 __getitem__ 中做复杂逻辑")

    # --- 方式 2：自定义 Dataset（灵活） ---
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            # 可以在这里做任何事情：数据增强、动态加载、条件过滤等
            x = self.X[idx]
            # 例如：给特征加噪声（数据增强）
            x = x + torch.randn_like(x) * 0.1
            return x, self.y[idx]

    custom_ds = MyDataset(X, y)
    print(f"\n自定义 Dataset:")
    print(f"  代码量: 需要实现 __len__ 和 __getitem__")
    print(f"  适用场景: 需要动态处理、数据增强、复杂逻辑")
    print(f"  优势: 完全控制数据加载过程")

    # 对比两者的输出
    print(f"\nTensorDataset 第 0 条 X 前 3 值: {tensor_ds[0][0][:3]}")
    print(f"自定义 Dataset 第 0 条 X 前 3 值: {custom_ds[0][0][:3]}")
    print("（自定义 Dataset 加了噪声，所以值不同）")


# ============================================================
# 五、实用技巧
# ============================================================

def demo_tips():
    print("\n" + "=" * 60)
    print("五、实用技巧")
    print("=" * 60)

    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)

    # 1. random_split 切分
    train_ds, val_ds, test_ds = random_split(dataset, [70, 15, 15])
    print(f"1. random_split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # 2. Subset 取子集
    from torch.utils.data import Subset
    subset = Subset(dataset, indices=[0, 1, 2, 3, 4])
    print(f"2. Subset (取前 5 条): {len(subset)}")

    # 3. DataLoader 常用参数
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,         # 每个 epoch 打乱顺序
        num_workers=0,        # 子进程数（0 = 主进程加载）
        pin_memory=True,      # GPU 训练时加速数据传输
        drop_last=True,       # 丢弃最后一个不完整的 batch
    )
    print(f"3. DataLoader (drop_last=True): {len(dataloader)} batches (100/16=6, 余 4 条丢弃)")

    # 4. 配合 GPU 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"4. 当前设备: {device}")
    print(f"   pin_memory=True 时，dataloader 输出的 tensor 可以直接 .to(device) 加速")

    # 5. 从 numpy 创建 TensorDataset
    import numpy as np
    np_data = np.random.randn(50, 8).astype(np.float32)
    np_labels = np.random.randint(0, 2, 50).astype(np.int64)
    tensor_from_numpy = TensorDataset(
        torch.from_numpy(np_data),
        torch.from_numpy(np_labels),
    )
    print(f"\n5. 从 numpy 创建: {len(tensor_from_numpy)} 条")


# ============================================================
# 运行所有演示
# ============================================================

if __name__ == "__main__":
    demo_basic()
    demo_training()
    demo_multi_input()
    demo_comparison()
    demo_tips()
