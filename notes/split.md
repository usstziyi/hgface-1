## 可以！有多种方式指定 train/test 比例

---

## 一、使用 `split` 参数（最常用）

### 1. 百分比分割
```python
from datasets import load_dataset

# 80% 训练，20% 测试
train_dataset = load_dataset("imdb", split="train[:80%]")
test_dataset = load_dataset("imdb", split="train[80%:]")

print(f"训练集: {len(train_dataset)}")  # 20000
print(f"测试集: {len(test_dataset)}")   # 5000
```

### 2. 绝对数量分割
```python
# 前10000条训练，后2000条验证
train_dataset = load_dataset("imdb", split="train[:10000]")
val_dataset = load_dataset("imdb", split="train[10000:12000]")

# 或指定负索引
test_dataset = load_dataset("imdb", split="train[-5000:]")  # 最后5000条
```

### 3. 组合分割
```python
# 多个分割组合
dataset = load_dataset("imdb", split=[
    "train[:80%]",      # 训练集
    "train[80%:90%]",   # 验证集  
    "train[90%:]"       # 测试集
])

train, val, test = dataset
print(len(train))  # 20000
print(len(val))    # 2500
print(len(test))   # 2500
```

---

## 二、更灵活的高级 split 语法

### 完整语法格式
```python
# 基本格式
"数据集名[选择1]+[选择2]"

# 具体例子
dataset = load_dataset("imdb", split="train[:50%]+train[-10%:]")
# 前50% + 最后10% = 60%的数据

dataset = load_dataset("imdb", split="train[:1000]+train[2000:3000]")
# 前1000条 + 第2000-3000条
```

### 交叉验证分割
```python
# k-fold 交叉验证（以5折为例）
k = 5
fold_size = 100 // k  # 20%

folds = []
for i in range(k):
    # 取第 i 折作为验证集
    val_split = f"train[{i*fold_size}%:{(i+1)*fold_size}%]"
    # 其他作为训练集
    train_split = f"train[:{i*fold_size}%]+train[{(i+1)*fold_size}%:]"
    
    fold_train = load_dataset("imdb", split=train_split)
    fold_val = load_dataset("imdb", split=val_split)
    folds.append((fold_train, fold_val))
```

---

## 三、使用 `train_test_split` 方法（更灵活）

```python
# 先加载完整数据，再分割
dataset = load_dataset("imdb", split="train")

# 方法1：简单比例分割
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(split_dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 20000})
#     test: Dataset({features: ['text', 'label'], num_rows: 5000})
# })

train = split_dataset["train"]
test = split_dataset["test"]
```

### 分层抽样（保持标签分布）
```python
# 按标签分层分割
split_dataset = dataset.train_test_split(
    test_size=0.2, 
    stratify_by_column="label",  # 按 label 分层
    seed=42
)

# 验证分布是否一致
import numpy as np
print("训练集正样本比例:", np.mean(split_dataset["train"]["label"]))
print("测试集正样本比例:", np.mean(split_dataset["test"]["label"]))
# 两者应该都接近 0.5
```

### 多部分分割（train/val/test）
```python
# 先分出测试集
train_test = dataset.train_test_split(test_size=0.2, seed=42)
# 再从训练集分出验证集
train_val = train_test["train"].train_test_split(test_size=0.1, seed=42)

# 最终得到
train = train_val["train"]          # 72%
val = train_val["test"]             # 8%  
test = train_test["test"]           # 20%
```

---

## 四、完整示例对比

### 示例1：加载时直接分割
```python
# 一行代码搞定 70/15/15 分割
dataset = load_dataset("imdb", split=[
    "train[:70%]",
    "train[70%:85%]", 
    "train[85%:]"
])
train, val, test = dataset
```

### 示例2：加载后分割（更灵活）
```python
# 先全部加载，需要时再分割
full_dataset = load_dataset("imdb", split="train")

# 70/15/15 分割
splits = full_dataset.train_test_split(test_size=0.3, seed=42)
val_test = splits["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = {
    "train": splits["train"],
    "val": val_test["train"],
    "test": val_test["test"]
}
```

---

## 五、split 参数完整语法表

| 语法 | 含义 | 示例 |
|------|------|------|
| `"train"` | 全部训练集 | `load_dataset("imdb", split="train")` |
| `"train[:1000]"` | 前1000条 | `split="train[:1000]"` |
| `"train[1000:]"` | 从第1000条到最后 | `split="train[1000:]"` |
| `"train[:50%]"` | 前50% | `split="train[:50%]"` |
| `"train[50%:]"` | 后50% | `split="train[50%:]"` |
| `"train[10%:30%]"` | 10%到30%区间 | `split="train[10%:30%]"` |
| `"train[:100]+train[-100:]"` | 前100 + 后100 | `split="train[:100]+train[-100:]"` |
| `["train[:80%]","train[80%:]"]` | 返回列表，80/20分割 | `split=["train[:80%]","train[80%:]"]` |

---

## 六、最佳实践建议

```python
# 🥇 推荐：固定随机种子，保证可重复性
dataset = load_dataset("imdb", split="train")
train, test = dataset.train_test_split(
    test_size=0.2, 
    stratify_by_column="label",  # 保持标签平衡
    seed=42  # 固定种子
).values()

# 🥈 简单场景：直接在 split 参数中指定
train = load_dataset("imdb", split="train[:80%]")
test = load_dataset("imdb", split="train[80%:]")
```

**总结**：两种方式都可以直接指定比例，`train_test_split` 更灵活（支持分层、打乱），`split` 参数更简洁快速。


## `dataset = load_dataset("imdb", split="train")` 详解

这行代码从 IMDB 数据集中**只加载训练集**，返回一个 `Dataset` 对象（不是 `DatasetDict`）。

---

## 一、与不加 split 的区别

### 不加 `split` 参数
```python
dataset = load_dataset("imdb")
print(type(dataset))
# <class 'datasets.dataset_dict.DatasetDict'>

print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
#     unsupervised: Dataset({features: ['text', 'label'], num_rows: 50000})
# })
```
返回的是 **DatasetDict**，需要再用 `dataset["train"]` 取子集。

### 加 `split="train"` 参数
```python
dataset = load_dataset("imdb", split="train")
print(type(dataset))
# <class 'datasets.arrow_dataset.Dataset'>

print(dataset)
# Dataset({features: ['text', 'label'], num_rows: 25000})
```
直接返回 **Dataset** 对象，可以直接用！

---

## 二、split 参数的本质

`split` 告诉 `load_dataset` **你要哪个数据子集**，而不是把整个数据集都下载下来。

```python
# IMDB 数据集有3个 split
load_dataset("imdb", split="train")        # 训练集 25000条
load_dataset("imdb", split="test")         # 测试集 25000条  
load_dataset("imdb", split="unsupervised") # 无标签 50000条
```

---

## 三、实际使用对比

### 不加 split（两步走）
```python
# 第一步：加载整个数据集
data = load_dataset("imdb")

# 第二步：取训练集
train_data = data["train"]
print(len(train_data))  # 25000
```

### 加 split（一步到位）
```python
# 一步：直接加载训练集
train_data = load_dataset("imdb", split="train")
print(len(train_data))  # 25000
```

---

## 四、为什么这样设计？

### 1. **节省内存**
```python
# 不加 split：下载全部3个部分（100,000条）
all_data = load_dataset("imdb")

# 加 split：只下载训练集（25,000条）
train = load_dataset("imdb", split="train")
```

### 2. **明确意图**
代码可读性更强，一看就知道只用了训练集。

### 3. **支持高级分割**
```python
# 可以直接指定百分比、数量
train = load_dataset("imdb", split="train[:80%]")
small = load_dataset("imdb", split="train[:1000]")
```

---

## 五、常见使用场景

### 场景1：只训练，不需要测试集
```python
# 只需训练集
train_dataset = load_dataset("imdb", split="train")
```

### 场景2：分别加载多个子集
```python
train = load_dataset("imdb", split="train")
test = load_dataset("imdb", split="test")
```

### 场景3：小样本调试
```python
# 只取100条快速测试代码
tiny_train = load_dataset("imdb", split="train[:100]")
tiny_test = load_dataset("imdb", split="test[:100]")
```

### 场景4：自定义分割比例
```python
# 从训练集切出验证集
train = load_dataset("imdb", split="train[:80%]")
val = load_dataset("imdb", split="train[80%:]")
```

---

## 六、完整对比表

| 写法 | 返回类型 | 数据量 | 后续访问 |
|------|---------|--------|---------|
| `load_dataset("imdb")` | `DatasetDict` | 全部100,000条 | `data["train"]` |
| `load_dataset("imdb", split="train")` | `Dataset` | 25,000条 | 直接使用 |
| `load_dataset("imdb", split="train[:100]")` | `Dataset` | 100条 | 直接使用 |
| `load_dataset("imdb", split=["train","test"])` | `List[Dataset]` | 50,000条 | `data[0]`, `data[1]` |

---

## 总结

**`split="train"` 就是告诉程序："我只要训练集，别给我加载其他的"**，返回一个可以直接迭代、处理的 `Dataset` 对象，省内存、代码更清晰。


## PyTorch 本身没有直接等价功能，但有多种替代方案

PyTorch 的 `torch.utils.data.Dataset` 是个**抽象接口**，不负责数据下载和分割管理。但可以通过以下方式实现类似功能：

---

## 一、PyTorch 内置的数据集子集工具

### 1. `Subset` - 取数据子集
```python
import torch
from torch.utils.data import Subset, DataLoader

# 假设你有一个完整数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

full_dataset = MyDataset(range(10000))

# 取前80% - 类似 load_dataset("imdb", split="train[:80%]")
indices = range(int(len(full_dataset) * 0.8))
train_subset = Subset(full_dataset, indices)

# 取后20%
val_indices = range(int(len(full_dataset) * 0.8), len(full_dataset))
val_subset = Subset(full_dataset, val_indices)

print(len(train_subset))  # 8000
print(len(val_subset))    # 2000
```

### 2. `random_split` - 随机分割数据集
```python
from torch.utils.data import random_split

# 最接近 HF split 的功能
full_dataset = MyDataset(range(10000))

# 80/20 随机分割
train_dataset, val_dataset = random_split(
    full_dataset, 
    [8000, 2000],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子
)

# 70/15/15 分割
train, val, test = random_split(
    full_dataset, 
    [7000, 1500, 1500],
    generator=torch.Generator().manual_seed(42)
)
```

---

## 二、完整对比：HF vs PyTorch

| 功能 | Hugging Face Datasets | PyTorch |
|------|----------------------|---------|
| **加载在线数据集** | ✅ `load_dataset("imdb")` | ❌ 需手动下载 |
| **按名称取子集** | ✅ `split="train"` | ❌ 无此概念 |
| **按百分比分割** | ✅ `split="train[:80%]"` | ⚠️ `random_split` + 手动计算 |
| **按数量分割** | ✅ `split="train[:1000]"` | ⚠️ `Subset` + 索引 |
| **分层抽样** | ✅ `stratify_by_column` | ❌ 需手动实现 |
| **返回类型** | `Dataset` 对象 | `Subset` 对象 |

---

## 三、PyTorch 中的实际解决方案

### 方案1：用 `torch.utils.data.random_split`
```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 创建数据集
texts = ["movie review " + str(i) for i in range(25000)]
labels = [i % 2 for i in range(25000)]
full_dataset = TextDataset(texts, labels)

# 分割 - 类似 HF 的 split 参数
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 创建 DataLoader
train_loader = DataLoader(train_set, batch_size=32)
val_loader = DataLoader(val_set, batch_size=32)
```

### 方案2：用 `Subset` 精确控制
```python
from torch.utils.data import Subset

# 按索引精确选择
train_indices = list(range(1000))  # 前1000条
val_indices = list(range(1000, 2000))  # 1000-2000条
test_indices = list(range(2000, 3000))  # 2000-3000条

train_set = Subset(full_dataset, train_indices)
val_set = Subset(full_dataset, val_indices)
test_set = Subset(full_dataset, test_indices)

# 类似 HF 的 split="train[:1000]"
train_small = Subset(full_dataset, range(1000))
```

### 方案3：使用 `torchvision` 的预定义数据集（如果适用）
```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# torchvision 数据集自带 train/test split
train_set = datasets.MNIST(
    root='./data', 
    train=True,   # 类似 HF 的 split="train"
    download=True
)

test_set = datasets.MNIST(
    root='./data', 
    train=False,  # 类似 HF 的 split="test"
    download=True
)
```

---

## 四、封装成类似 HF 的接口

```python
class DatasetSplitter:
    """模拟 Hugging Face 的 split 功能"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.total_size = len(dataset)
    
    def split(self, spec):
        """支持类似 HF 的 split 语法"""
        if isinstance(spec, list):
            return [self._parse_split(s) for s in spec]
        return self._parse_split(spec)
    
    def _parse_split(self, spec):
        if ':' not in spec:
            return self.dataset
        
        # 解析 "train[:80%]" 格式
        part = spec.split(':')[1]
        
        if '%' in part:
            if part.endswith('%]'):
                # [:80%]
                percent = int(part[:-2])
                size = int(self.total_size * percent / 100)
                return Subset(self.dataset, range(size))
            elif part.endswith('%:]'):
                # [80%:]
                percent = int(part[:-3])
                start = int(self.total_size * percent / 100)
                return Subset(self.dataset, range(start, self.total_size))
        else:
            # [1000:2000]
            if ':' in part:
                start, end = part.split(':')
                start = int(start) if start else 0
                end = int(end) if end else self.total_size
                return Subset(self.dataset, range(start, end))
            else:
                # [:1000]
                size = int(part)
                return Subset(self.dataset, range(size))

# 使用示例
full_dataset = TextDataset(texts, labels)
splitter = DatasetSplitter(full_dataset)

# 类似 HF 的用法
train = splitter.split("train[:80%]")
val = splitter.split("train[80%:]")
small = splitter.split("train[:1000]")
```

---

## 五、PyTorch 生态的替代品

### 1. **PyTorch Lightning DataModule**
```python
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        full_data = MyDataset(data, labels)
        
        # 分割
        self.train, self.val, self.test = random_split(
            full_data, [7000, 1500, 1500]
        )
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)
```

### 2. **结合 HF datasets（推荐）**
```python
# 最佳实践：用 HF 加载和处理数据，转 PyTorch 格式
from datasets import load_dataset

# HF 负责数据加载和分割
dataset = load_dataset("imdb", split="train[:80%]")
dataset.set_format(type="torch", columns=["input_ids", "label"])

# PyTorch 负责训练
loader = DataLoader(dataset, batch_size=32)
```

---

## 总结

**PyTorch 本身**没有像 HF 那样便捷的 `split` 语法，但提供了：
- `random_split`：随机分割（最接近 HF 功能）
- `Subset`：按索引取子集
- `torchvision` 数据集：预定义 train/test split

**推荐做法**：
- 简单项目：用 `random_split` + `Subset`
- 复杂数据：直接用 **Hugging Face datasets** 处理，然后转 PyTorch 格式
- 这样既享受 HF 的便捷，又能用 PyTorch 训练