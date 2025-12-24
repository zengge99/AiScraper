import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SimpleMediaDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_path_len=128, max_name_len=32):
        self.tokenizer = tokenizer
        self.max_path_len = max_path_len
        self.max_name_len = max_name_len
        # 加载数据：只取第一个#后的内容作为名称
        self.data = self._load_data(data_file)

    def _load_data(self, data_file):
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                # 仅按第一个#拆分，忽略后续所有#
                if "#" not in line:
                    print(f"警告：第{line_num}行无#分隔符，跳过该行：{line}")
                    continue
                # split("#", 1)：只拆分第一个#，返回[路径, 名称+后续内容]
                path_part, name_part = line.split("#", 1)
                # 名称仅取第一个#后的内容（自动忽略后续#及内容）
                media_name = name_part.strip()
                # 过滤空名称
                if not media_name:
                    print(f"警告：第{line_num}行名称为空，跳过该行：{line}")
                    continue
                data.append({"path": path_part.strip(), "name": media_name})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = item["path"]
        name = item["name"]

        # 编码路径
        path_encoding = self.tokenizer(
            path,
            max_length=self.max_path_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            use_fast=False
        )
        # 编码名称
        name_encoding = self.tokenizer(
            name,
            max_length=self.max_name_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            use_fast=False
        )

        return {
            "path_ids": path_encoding["input_ids"].squeeze(),
            "path_mask": path_encoding["attention_mask"].squeeze(),
            "name_ids": name_encoding["input_ids"].squeeze(),
            "name_mask": name_encoding["attention_mask"].squeeze()
        }