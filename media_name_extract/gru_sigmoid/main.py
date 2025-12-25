import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pickle
import sys
import os
import re

# --- 全局核心配置 ---
NUM_THREADS = 4
BATCH_SIZE = 64
LR = 1e-4            # 学习率
EPOCHS = 50          # 训练轮数
MAX_LEN = 150        # 最大路径长度
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
DATA_FILE = "train_data.txt"

# --- 预测/调试配置 ---
DEBUG_MODE = True    # 开启后显示全路径所有字符得分
THRESHOLD = 0.2      # 核心判定阈值
SMOOTH_VAL = 0.05    # 辅助判定阈值（用于救回中间字符）

# 必须在 import torch 之后立即设置
torch.set_num_threads(NUM_THREADS)

# --- 模型结构定义 ---
class FilmExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        return self.fc(gru_out).squeeze(-1)

# --- 数据集定义 ---
class MovieDataset(Dataset):
    def __init__(self, lines, char_to_idx, max_len=MAX_LEN):
        self.samples = []
        skipped_count = 0
        
        for line in lines:
            line = line.strip()
            if '#' not in line: continue
            input_path, target_name = line.rsplit('#', 1)
            target_name = target_name.strip()
            
            # 构造正则：允许目标名中的空格对应路径中的 "." "_" 或 " "
            # 例如目标 "Transformers The" -> 正则 "Transformers[._\s]+The"
            escaped_target = re.escape(target_name)
            pattern = escaped_target.replace(r'\ ', r'[._\s]+')
            
            # 在路径中搜索匹配项 (忽略大小写)
            match = re.search(pattern, input_path, re.IGNORECASE)
            
            if match:
                start_idx = match.start()
                end_idx = match.end()
                
                # 构建 Label
                input_ids = [char_to_idx.get(c, 1) for c in input_path[:max_len]]
                labels = [0.0] * len(input_ids)
                
                # 只有匹配到的部分标为 1.0
                limit = min(end_idx, max_len)
                for i in range(start_idx, limit):
                    labels[i] = 1.0
                
                # Padding
                pad_len = max_len - len(input_ids)
                self.samples.append((
                    torch.tensor(input_ids + [0] * pad_len), 
                    torch.tensor(labels + [0.0] * pad_len)
                ))
            else:
                # 如果完全匹配不到（比如数据标注错了），则跳过
                skipped_count += 1
            # --- 核心修改结束 ---

        if skipped_count > 0:
            print(f"⚠️ 跳过了 {skipped_count} 条无法匹配标签的数据。")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- 训练逻辑 ---
def run_train():
    if not os.path.exists(DATA_FILE): 
        print(f"❌ 找不到数据文件 {DATA_FILE}"); return
        
    with open(DATA_FILE, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
    
    # 构建词表
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
        print("ℹ️ 已加载现有词表。")
    else:
        # 只统计 '#' 左边的字符（即路径部分）
        raw_paths = [l.split('#')[0] for l in lines if '#' in l]
        all_chars = set("".join(raw_paths))
        char_to_idx = {c: i+2 for i, c in enumerate(sorted(list(all_chars)))}
        char_to_idx['<PAD>'], char_to_idx['<UNK>'] = 0, 1
        with open(VOCAB_PATH, 'wb') as f: pickle.dump(char_to_idx, f)
        print(f"🆕 已创建新词表，包含 {len(char_to_idx)} 个字符。")

    dataset = MovieDataset(lines, char_to_idx)
    if len(dataset) < 2:
        print("❌ 有效样本数量不足，无法进行训练。"); return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # 修复：防止数据集过小导致 val_size 为 0
    if val_size < 1: train_size -= 1; val_size += 1
        
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FilmExtractor(len(char_to_idx))
    if os.path.exists(MODEL_PATH):
        print(f"🔄 检测到现有模型，加载权重以 LR={LR} 继续微调...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    best_val_loss = float('inf') 

    print(f"🚀 开始训练 | 样本数: {len(dataset)} (Skip失败样本)")
    try:
        for epoch in range(EPOCHS):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}")
            for x, y in pbar:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    v_loss += criterion(model(vx), vy).item()
            
            avg_val_loss = v_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f" ✨ 验证集 Loss 提升至 {avg_val_loss:.4f}，模型已保存。")
            else:
                print(f" ⏳ 验证集 Loss: {avg_val_loss:.4f} (未提升)")
    except KeyboardInterrupt: print("\n🛑 用户手动停止训练。")

# --- 预测逻辑 (核心修改：后处理) ---
def run_predict(path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("❌ 错误: 找不到模型或词表文件。请先运行训练。"); return

    with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
    model = FilmExtractor(len(char_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # 预处理：截断和转换 ID
    input_ids = [char_to_idx.get(c, 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    
    with torch.no_grad():
        probs = model(torch.tensor([padded]))[0][:len(path)].numpy()

    if DEBUG_MODE:
        print(f"\n{'='*65}")
        print(f"{'索引':<4} | {'字符':<4} | {'分值 (10位小数)':<15} | 状态")
        print("-" * 65)
        for i, p in enumerate(probs):
            status = "✅ [选中]" if p > THRESHOLD else "   [排除]"
            print(f"{i:<4} | {path[i]:<4} | {p:.10f} | {status}")
        print(f"{'='*65}\n")

    # 增强型提取逻辑 (桥梁逻辑)
    res_list = []
    for i, p in enumerate(probs):
        is_high = p > THRESHOLD
        is_bridge = False
        if not is_high and p > SMOOTH_VAL:
            left_high = probs[i-1] > THRESHOLD if i > 0 else False
            right_high = probs[i+1] > THRESHOLD if i < len(probs)-1 else False
            if left_high and right_high:
                is_bridge = True
        
        if is_high or is_bridge:
            res_list.append(path[i])
    
    raw_result = "".join(res_list)
    
    # 目的：将 Transformers.The.Last.Knight 转换为 Transformers The Last Knight
    # A. 替换点和下划线为空格
    clean_result = raw_result.replace('.', ' ').replace('_', ' ')
    
    # B. 再次正则清洗：把连续的空格变成单个空格
    clean_result = re.sub(r'\s+', ' ', clean_result)
    
    # C. 去掉首尾可能残留的非字母符号 (如 / ( ) - 等)
    clean_result = clean_result.strip("/()# “”.-")

    if DEBUG_MODE: 
        print(f"📥 提取原文: {raw_result}")
        print(f"✅ 最终结果: {clean_result}\n")
    else: 
        print(clean_result)

# --- 入口控制 ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_predict(sys.argv[1])
    else:
        # 不带参数：训练模式
        run_train()