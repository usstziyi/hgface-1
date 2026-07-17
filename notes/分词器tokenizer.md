原始文本："I love movies!"
    ↓
① 分词 (Tokenization)
    ↓
['I', 'love', 'movies', '!']
    ↓
② 转换为ID (Convert to IDs)
    ↓
[1045, 2293, 6284, 999]
    ↓
③ 添加特殊标记 (Add Special Tokens)
    ↓
[101, 1045, 2293, 6284, 999, 102]  (BERT: [CLS] ... [SEP])
    ↓
④ 填充/截断 (Pad/Truncate)
    ↓
[101, 1045, 2293, 6284, 999, 102, 0, 0, ..., 0]  (填充到固定长度)
    ↓
⑤ 创建 attention mask
    ↓
[1, 1, 1, 1, 1, 1, 0, 0, ..., 0]  (1=真实token, 0=填充)
    ↓
最终输出：模型可以直接使用的数字序列