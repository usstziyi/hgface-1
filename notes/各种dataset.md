
### Python、PyTorch、Hugging Face 的 Dataset 对比表

| 维度 | 🐍 **Python 原生类型** <br>（`list` / `dict`） | 🔥 **PyTorch** <br>`torch.utils.data.Dataset` | 🤗 **Hugging Face Datasets** <br>`datasets.Dataset` |
|------|-----------------------------------------------|---------------------------------------------|---------------------------------------------------|
| **本质** | 内置数据结构，存储 Python 对象 | 抽象类（接口），需用户继承实现 | 高级表格容器，底层是 Apache Arrow 内存映射表 |
| **继承/关系** | 不继承任何特殊类 | 继承 `torch.utils.data.Dataset` | 独立实现的类 `datasets.arrow_dataset.Dataset` |
| **是否为 dict** | `dict` 是；`list` 不是 | ❌ 不是 | ❌ 不是（但 `DatasetDict` 容纳多个 split） |
| **核心功能** | 存储和访问数据 | 定义获取样本的标准接口 | 高效存储、处理、流式访问大规模数据集 |
| **必须实现的方法** | 无（直接使用） | `__len__` 和 `__getitem__` | 无需实现（库已封装好） |
| **典型访问方式** | `data[key]` 或 `data[i]` | `dataset[i]` 返回第 i 个样本 | `dataset[i]` 返回整行；`dataset["col"]` 返回整列 |
| **按列名访问** | `dict` 可以；`list` 不行 | ❌ 不支持（索引只返回一个样本） | ✅ 支持（`dataset["text"]` 返回所有文本） |
| **按切片访问** | `list` 支持；`dict` 不支持 | ✅ 支持（`dataset[:5]` 返回子集） | ✅ 支持（`dataset[:5]` 返回子 Dataset） |
| **返回数据类型** | Python 对象 | 通常返回 `torch.Tensor`（由用户定义） | Python 字典或通过 `set_format` 指定为张量 |
| **内存效率** | 全加载到内存，开销大 | 全加载，靠 `DataLoader` 分批 | 内存映射（memory-mapped），不一次性读入内存 |
| **流式处理** | ❌ 不支持 | ❌ 不支持（需自己写） | ✅ `streaming=True` 可迭代加载 |
| **内置处理/映射** | 用推导式或循环手动处理 | 无，靠 `DataLoader` + `collate_fn` | ✅ `map()`、`filter()`、`shuffle()` 等内置方法 |
| **主要用途** | 小规模数据、快速原型、作为输入源 | 作为 `DataLoader` 的输入，供模型训练/推理用 | 加载、预处理大规模 NLP/CV 数据集 |
| **生态与存储** | 内存、JSON/Pickle 文件 | 与 `DataLoader` 紧耦合 | 存储在磁盘/缓存，支持 Hugging Face Hub 直读 |

