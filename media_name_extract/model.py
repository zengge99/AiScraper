import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class MediaNameExtractor(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-chinese"):
        super().__init__()
        # 加载预训练BERT模型（消除torch_dtype警告，改用dtype）
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        # 降低Dropout强度，先让模型学到特征
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        # 加载配套分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, path_ids, path_mask, name_ids=None, name_mask=None):
        # 编码路径文本
        path_bert_out = self.bert(input_ids=path_ids, attention_mask=path_mask)
        path_emb = path_bert_out.last_hidden_state  # [batch, path_len, 768]
        path_emb = self.dropout(path_emb)

        if name_ids is not None and name_mask is not None:
            # 训练阶段：编码名称文本+计算相似度
            name_bert_out = self.bert(input_ids=name_ids, attention_mask=name_mask)
            name_emb = name_bert_out.last_hidden_state  # [batch, name_len, 768]
            name_emb = self.dropout(name_emb)

            # 归一化+计算余弦相似度矩阵
            path_emb_norm = F.normalize(path_emb, dim=-1)
            name_emb_norm = F.normalize(name_emb, dim=-1)
            similarity_matrix = torch.matmul(path_emb_norm, name_emb_norm.transpose(1, 2))  # [batch, path_len, name_len]
            
            # 取最大相似度+缩放分数范围（核心优化：放大差异）
            max_similarity = torch.max(similarity_matrix, dim=-1)[0]  # [batch, path_len]
            max_similarity = max_similarity * 5  # 从[-1,1]缩放至[-5,5]，放大Loss优化空间

            # 生成精准标签（强化正负样本信号）
            batch_size, path_len = max_similarity.shape
            label = torch.zeros_like(max_similarity, device=max_similarity.device)
            for b in range(batch_size):
                # 提取名称文本
                name_tokens = self.tokenizer.convert_ids_to_tokens(name_ids[b])
                name_tokens = [t for t in name_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
                name_str = "".join(name_tokens)
                # 提取路径文本
                path_tokens = self.tokenizer.convert_ids_to_tokens(path_ids[b])
                path_str = "".join([t for t in path_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]])
                
                # 精准标注名称字符为1，其余为0
                if name_str in path_str:
                    start_idx = path_str.find(name_str)
                    end_idx = start_idx + len(name_str)
                    for i in range(start_idx, end_idx):
                        if i < path_len:
                            label[b][i] = 1.0
                # 强制冗余词标0
                redundant_tokens = {"第", "季", "集", "4K", "HDR", "中英双字", "mp4", "mkv", "美", "剧"}
                for i, token in enumerate(path_tokens):
                    if token in redundant_tokens and i < path_len:
                        label[b][i] = 0.0

            # BCEWithLogitsLoss（适配缩放后的分数）
            loss = F.binary_cross_entropy_with_logits(max_similarity, label)
            return {"loss": loss, "similarity": max_similarity}
        else:
            # 推理阶段：仅编码路径+输出分数
            path_emb_norm = F.normalize(path_emb, dim=-1)
            token_scores = self.fc(path_emb_norm).squeeze(-1)  # [1, path_len]
            token_scores = torch.sigmoid(token_scores)  # 归一化到[0,1]
            return {"token_scores": token_scores}

    @torch.no_grad()
    def extract_name(self, path):
        # 文本编码
        encoding = self.tokenizer(
            path,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 设备对齐
        device = next(self.parameters()).device
        path_ids = encoding["input_ids"].to(device)
        path_mask = encoding["attention_mask"].to(device)

        # 前向推理
        outputs = self.forward(path_ids, path_mask)
        scores = outputs["token_scores"].squeeze().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(path_ids[0])

        # 打印分数详情
        print("\n=== 字符分数详情（0~1）===")
        for token, score in zip(tokens, scores):
            if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                print(f"字符：{token:<4} | 分数：{score:.4f}")

        # 筛选有效字符（简化逻辑+固定阈值）
        name_chars = []
        # 扩展冗余词列表
        strong_redundant = {
            "/", "\\", ".", "（", "）", "(", ")", " ", 
            "4K", "HDR", "中英双字", "国语", "中字", "蓝光", "原盘",
            "mp4", "mkv", "avi", "flv", "第", "季", "集", "全", "美", "剧"
        }
        # 固定阈值（略高于0.5）
        threshold = 0.51

        for token, score in zip(tokens, scores):
            if (score > threshold) and \
               (token not in strong_redundant) and \
               (token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]) and \
               token.strip():
                name_chars.append(token)

        # 去重并拼接结果
        name_chars_unique = list(dict.fromkeys(name_chars))
        name = "".join(name_chars_unique).strip()

        if not name or name.isdigit():
            print(f"\n最终结果：未识别到影视名称（有效字符：{name_chars_unique}）")
            return "未识别到影视名称"
        print(f"\n最终结果：{name}")
        return name