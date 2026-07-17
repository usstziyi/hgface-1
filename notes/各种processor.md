```python
# 纯文本处理器
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 只有 tokenizer

# 纯图像处理器
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
# 只有 image_processor

# CLIP Processor（文本 + 图像）
from transformers import CLIPProcessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer + image_processor
```