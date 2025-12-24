import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SimpleMediaDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_path_len=128, max_name_len=32):
        self.tokenizer = tokenizer
        self.max_path_len = max_path_len
        self.max_name_len = max_name_len
        self.data = self._load_and_validate_data(data_file)

    def _load_and_validate_data(self, data_file):
        """加载并校验数据，过滤无效样本"""
        valid_data = []
        with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # 拆分路径和名称（至少包含一个#）
                parts = line.split("#", 2)
                if len(parts) < 2:
                    print(f"警告：第{line_num}行格式错误，缺少#分隔符，已跳过")
                    continue
                path = parts[0].strip()
                name = parts[1].strip()
                if not path or not name:
                    print(f"警告：第{line_num}行路径/名称为空，已跳过")
                    continue
                valid_data.append((path, name))
        print(f"数据加载完成：共{len(valid_data)}条有效样本")
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, name = self.data[idx]
        
        # 编码路径
        path_encoding = self.tokenizer(
            path,
            max_length=self.max_path_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 编码名称
        name_encoding = self.tokenizer(
            name,
            max_length=self.max_name_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 挤压维度（去掉batch维度）
        return {
            "path_ids": path_encoding["input_ids"].squeeze(0),
            "path_mask": path_encoding["attention_mask"].squeeze(0),
            "name_ids": name_encoding["input_ids"].squeeze(0),
            "name_mask": name_encoding["attention_mask"].squeeze(0)
        }