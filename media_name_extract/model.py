import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class MediaNameExtractor(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        # 删除use_fast=False参数
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # 名称提取头：生成候选名称的相似度分数
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, path_ids, path_mask, name_ids=None, name_mask=None):
        """
        path_ids/path_mask：文件路径的编码
        name_ids/name_mask：目标名称的编码（训练时传入，推理时无）
        """
        # 1. 编码路径文本
        path_emb = self.bert(input_ids=path_ids, attention_mask=path_mask).last_hidden_state  # [batch, path_len, 768]
        path_emb = self.dropout(path_emb)

        if name_ids is not None:
            # 训练阶段：计算路径中每个token与目标名称的匹配损失
            name_emb = self.bert(input_ids=name_ids, attention_mask=name_mask).last_hidden_state  # [batch, name_len, 768]
            name_emb = self.dropout(name_emb)
            
            # 计算路径token与名称token的余弦相似度
            path_emb_norm = F.normalize(path_emb, dim=-1)
            name_emb_norm = F.normalize(name_emb, dim=-1)
            similarity = torch.matmul(path_emb_norm, name_emb_norm.transpose(1, 2))  # [batch, path_len, name_len]
            max_similarity = torch.max(similarity, dim=-1)[0]  # [batch, path_len]：每个路径token与名称的最大相似度
            
            # 损失：让名称对应的token相似度趋近1，其余趋近0
            loss = self._compute_matching_loss(path_ids, name_ids, max_similarity)
            return {"loss": loss, "similarity": max_similarity}
        else:
            # 推理阶段：输出路径中每个token的重要性分数
            token_scores = self.fc(path_emb).squeeze(-1)  # [batch, path_len]
            return {"token_scores": token_scores}

    def _compute_matching_loss(self, path_ids, name_ids, max_similarity):
        """计算匹配损失：名称在路径中的token分数→1，其余→0"""
        batch_size = path_ids.shape[0]
        loss = 0.0

        for b in range(batch_size):
            # 把路径和名称转成字符串（去特殊token）
            path_str = self.tokenizer.decode(path_ids[b], skip_special_tokens=True)
            name_str = self.tokenizer.decode(name_ids[b], skip_special_tokens=True)
            
            # 找到名称在路径中的起始/结束索引
            if name_str in path_str:
                start_idx = path_str.index(name_str)
                end_idx = start_idx + len(name_str)
            else:
                # 名称不在路径中时，用模糊匹配（取最相似的片段）
                start_idx, end_idx = 0, len(name_str)

            # 生成监督标签：名称区间内=1，其余=0
            path_len = path_ids[b].shape[0]
            label = torch.zeros(path_len).to(path_ids.device)
            # 映射字符索引到token索引（中文按字符拆分，一一对应）
            for i in range(path_len):
                token = self.tokenizer.decode([path_ids[b][i]])
                if start_idx <= i < end_idx and token.strip() != "":
                    label[i] = 1.0

            # 计算MSE损失（相似度分数趋近标签值）
            loss += F.mse_loss(max_similarity[b], label)

        return loss / batch_size

    """ @torch.no_grad()
    def extract_name(self, path):
        """推理：从路径提取名称"""
        # 编码路径（删除use_fast=False）
        encoding = self.tokenizer(
            path,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        path_ids = encoding["input_ids"].to(next(self.parameters()).device)
        path_mask = encoding["attention_mask"].to(next(self.parameters()).device)

        # 预测每个token的重要性分数
        outputs = self.forward(path_ids, path_mask)
        scores = outputs["token_scores"].squeeze().cpu().numpy()

        # 提取高分token（分数>0.5的视为名称字符）
        tokens = self.tokenizer.convert_ids_to_tokens(path_ids.squeeze().cpu().numpy())
        name_chars = []
        for token, score in zip(tokens, scores):
            if score > 0.5 and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                # 过滤冗余符号/关键词
                if token not in ["/", "\\", ".", "（", "）", "(", ")", " ", "4K", "1080P", "HDR", "国语", "中字", "超清", "蓝光", "原盘", "系列", "部", "集"]:
                    name_chars.append(token)

        # 去重+拼接+清理
        name = "".join(list(dict.fromkeys(name_chars))).strip()
        # 过滤纯数字/空字符串
        if not name or name.isdigit():
            return "未识别到影视名称"
        return name """

    # 在 model.py 的 extract_name 函数中，添加打印逻辑（临时调试）
    @torch.no_grad()
    def extract_name(self, path):
        encoding = self.tokenizer(
            path,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        path_ids = encoding["input_ids"].to(next(self.parameters()).device)
        path_mask = encoding["attention_mask"].to(next(self.parameters()).device)

        outputs = self.forward(path_ids, path_mask)
        scores = outputs["token_scores"].squeeze().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(path_ids.squeeze().cpu().numpy())

        # 新增：打印所有字符和对应分数（关键调试信息）
        print("\n=== 字符分数详情 ===")
        for token, score in zip(tokens, scores):
            if token not in ["[PAD]", "[CLS]", "[SEP]"]:  # 过滤特殊符号
                print(f"字符：{token} | 分数：{score:.4f}")

        # 原有过滤逻辑
        name_chars = []
        for token, score in zip(tokens, scores):
            if score > 0.5 and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                if token not in ["/", "\\", ".", "（", "）", "(", ")", " ", "4K", "1080P", "HDR", "国语", "中字", "超清", "蓝光", "原盘", "系列", "部", "集"]:
                    name_chars.append(token)

        name = "".join(list(dict.fromkeys(name_chars))).strip()
        if not name or name.isdigit():
            print(f"\n最终结果：未识别到影视名称（有效字符：{name_chars}）")
            return "未识别到影视名称"
        return name