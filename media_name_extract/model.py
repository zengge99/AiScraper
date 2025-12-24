import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class MediaNameExtractor(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # 提高Dropout概率，增强正则化
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, path_ids, path_mask, name_ids=None, name_mask=None):
        """
        优化点：
        1. 更换损失函数为 BCEWithLogitsLoss
        2. 优化标签生成逻辑，排除路径中的冗余层级
        """
        path_bert_out = self.bert(input_ids=path_ids, attention_mask=path_mask)
        path_emb = path_bert_out.last_hidden_state
        path_emb = self.dropout(path_emb)

        if name_ids is not None:
            # 训练阶段
            name_bert_out = self.bert(input_ids=name_ids, attention_mask=name_mask)
            name_emb = name_bert_out.last_hidden_state
            name_emb = self.dropout(name_emb)

            # 余弦相似度计算
            path_emb_norm = F.normalize(path_emb, dim=-1)
            name_emb_norm = F.normalize(name_emb, dim=-1)
            similarity_matrix = torch.matmul(path_emb_norm, name_emb_norm.transpose(1, 2))
            max_similarity = torch.max(similarity_matrix, dim=-1)[0]

            # ========== 优化标签生成逻辑 ==========
            batch_size, path_len = max_similarity.shape
            label = torch.zeros_like(max_similarity, device=max_similarity.device)
            
            for b in range(batch_size):
                # 提取名称的有效字符（去特殊符号）
                name_len = torch.sum(name_mask[b]).item()
                name_ids_valid = name_ids[b][:name_len]
                name_tokens = self.tokenizer.convert_ids_to_tokens(name_ids_valid)
                name_tokens = [t for t in name_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
                name_str = "".join(name_tokens)

                # 提取路径的有效字符
                path_tokens = self.tokenizer.convert_ids_to_tokens(path_ids[b])
                
                # 关键优化：只标注名称完整匹配的字符，排除路径层级/冗余词
                path_str = "".join([t for t in path_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]])
                if name_str in path_str:
                    # 找到名称在路径中的起始位置
                    start_idx = path_str.find(name_str)
                    end_idx = start_idx + len(name_str)
                    # 只对名称对应的字符标1
                    for i in range(start_idx, end_idx):
                        if i < len(path_tokens):
                            label[b][i] = 1.0

            # ========== 更换为 BCEWithLogitsLoss ==========
            # 该损失对正负样本区分更敏感，避免特征塌陷
            loss = F.binary_cross_entropy_with_logits(max_similarity, label)
            return {"loss": loss, "similarity": max_similarity}
        else:
            # 推理阶段：Sigmoid归一化到0~1
            path_emb_norm = F.normalize(path_emb, dim=-1)
            token_scores = self.fc(path_emb_norm).squeeze(-1)
            token_scores = torch.sigmoid(token_scores)
            return {"token_scores": token_scores}

    @torch.no_grad()
    def extract_name(self, path):
        """
        优化点：
        1. 相对阈值筛选（基于分数均值）
        2. 扩展冗余词过滤列表
        3. 过滤特殊符号 [UNK]/[CLS] 等
        """
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

        # 打印分数详情
        print("\n=== 字符分数详情（0~1）===")
        for token, score in zip(tokens, scores):
            if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                print(f"字符：{token:<4} | 分数：{score:.4f}")

        # ========== 优化筛选逻辑 ==========
        name_chars = []
        # 扩展强冗余词列表
        strong_redundant = {
            "/", "\\", ".", "（", "）", "(", ")", " ", 
            "4K", "HDR", "中英双字", "国语", "中字", "蓝光", "原盘",
            "mp4", "mkv", "avi", "flv", "第", "季", "集", "全"
        }
        # 计算分数均值，取高于均值+0.01的字符（相对阈值，适配不同样本）
        valid_scores = [s for t, s in zip(tokens, scores) if t not in ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]]
        score_mean = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
        threshold = score_mean + 0.01

        for token, score in zip(tokens, scores):
            if (score > threshold) and \
               (token not in strong_redundant) and \
               (token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]) and \
               token.strip():
                name_chars.append(token)

        # 去重并拼接
        name_chars_unique = list(dict.fromkeys(name_chars))
        name = "".join(name_chars_unique).strip()

        if not name or name.isdigit():
            print(f"\n最终结果：未识别到影视名称（有效字符：{name_chars_unique}）")
            return "未识别到影视名称"
        print(f"\n最终结果：{name}")
        return name