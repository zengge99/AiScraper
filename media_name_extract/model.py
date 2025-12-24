import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class MediaNameExtractor(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-chinese"):
        super().__init__()
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # dropout层缓解过拟合
        self.dropout = nn.Dropout(0.3)
        # 线性层：将768维向量映射到1维分数
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, path_ids, path_mask, name_ids=None, name_mask=None):
        """
        统一训练/推理逻辑：
        - 训练：输入路径+名称，计算相似度损失
        - 推理：仅输入路径，输出0~1的名称概率分数
        """
        # 1. BERT编码路径文本（核心语义提取）
        path_bert_out = self.bert(input_ids=path_ids, attention_mask=path_mask)
        path_emb = path_bert_out.last_hidden_state  # [batch, seq_len, 768]
        path_emb = self.dropout(path_emb)

        if name_ids is not None:
            # ========== 训练阶段逻辑 ==========
            # 编码名称文本
            name_bert_out = self.bert(input_ids=name_ids, attention_mask=name_mask)
            name_emb = name_bert_out.last_hidden_state  # [batch, name_len, 768]
            name_emb = self.dropout(name_emb)

            # 余弦相似度计算（归一化到0~1）
            path_emb_norm = F.normalize(path_emb, dim=-1)  # [batch, path_len, 768]
            name_emb_norm = F.normalize(name_emb, dim=-1)  # [batch, name_len, 768]
            
            # 计算路径字符与名称字符的相似度矩阵 [batch, path_len, name_len]
            similarity_matrix = torch.matmul(path_emb_norm, name_emb_norm.transpose(1, 2))
            # 每个路径字符取与名称字符的最大相似度 [batch, path_len]
            max_similarity = torch.max(similarity_matrix, dim=-1)[0]

            # 生成标签：名称字符在路径中的位置为1，其他为0
            batch_size, path_len = max_similarity.shape
            label = torch.zeros_like(max_similarity, device=max_similarity.device)
            
            for b in range(batch_size):
                # 提取名称的有效字符（排除padding和特殊符号）
                name_len = torch.sum(name_mask[b]).item()
                name_ids_valid = name_ids[b][:name_len]
                name_tokens = self.tokenizer.convert_ids_to_tokens(name_ids_valid)
                name_tokens = [t for t in name_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
                
                # 提取路径的有效字符
                path_ids_valid = path_ids[b]
                path_tokens = self.tokenizer.convert_ids_to_tokens(path_ids_valid)
                
                # 标注：路径中出现的名称字符设为1
                for i, path_token in enumerate(path_tokens):
                    if path_token in name_tokens and path_token not in ["[CLS]", "[SEP]", "[PAD]"]:
                        label[b][i] = 1.0

            # 计算MSE损失（目标是让相似度分数接近标签1/0）
            loss = F.mse_loss(max_similarity, label)
            return {"loss": loss, "similarity": max_similarity}
        else:
            # ========== 推理阶段逻辑 ==========
            # 归一化路径向量 + 线性层映射 + Sigmoid归一化到0~1
            path_emb_norm = F.normalize(path_emb, dim=-1)
            token_scores = self.fc(path_emb_norm).squeeze(-1)  # [batch, path_len]
            token_scores = torch.sigmoid(token_scores)  # 关键：映射到0~1区间
            return {"token_scores": token_scores}

    @torch.no_grad()
    def extract_name(self, path):
        """推理：从路径提取影视名称"""
        # 1. 文本编码（适配BERT输入格式）
        encoding = self.tokenizer(
            path,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        path_ids = encoding["input_ids"].to(next(self.parameters()).device)
        path_mask = encoding["attention_mask"].to(next(self.parameters()).device)

        # 2. 模型预测（0~1分数）
        outputs = self.forward(path_ids, path_mask)
        scores = outputs["token_scores"].squeeze().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(path_ids.squeeze().cpu().numpy())

        # 3. 打印调试信息（查看分数分布）
        print("\n=== 字符分数详情（0~1）===")
        for token, score in zip(tokens, scores):
            if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                print(f"字符：{token:<4} | 分数：{score:.4f}")

        # 4. 筛选有效字符（降低阈值，精简过滤规则）
        name_chars = []
        redundant_tokens = {"/", "\\", ".", "（", "）", "(", ")", " "}
        for token, score in zip(tokens, scores):
            # 分数阈值调低到0.2，仅过滤绝对冗余符号
            if score > 0.2 and token not in redundant_tokens and token.strip():
                name_chars.append(token)

        # 5. 去重+拼接结果
        # 去重但保留顺序：dict.fromkeys可以保留首次出现的顺序
        name_chars_unique = list(dict.fromkeys(name_chars))
        name = "".join(name_chars_unique).strip()

        # 6. 结果校验
        if not name or name.isdigit():
            print(f"\n最终结果：未识别到影视名称（有效字符：{name_chars_unique}）")
            return "未识别到影视名称"
        print(f"\n最终结果：{name}")
        return name