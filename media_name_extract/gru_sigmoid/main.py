import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pickle
import sys
import os
import re
import random
import numpy as np  # 引入numpy用于固定种子

# --- 全局核心配置 ---
NUM_THREADS = 4
BATCH_SIZE = 64
LR = 1e-4            # 学习率
EPOCHS = 50          # 训练轮数
MAX_LEN = 150        # 最大路径长度
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
DATA_FILE = "train_data.txt"
SEED = 42            # 🎲 固定随机种子

# --- 预测/调试配置 ---
DEBUG_MODE = True    # 开启后显示全路径所有字符得分
THRESHOLD = 0.2      # 核心判定阈值
SMOOTH_VAL = 0.05    # 辅助判定阈值（用于救回中间字符）

# 必须在 import torch 之后立即设置
torch.set_num_threads(NUM_THREADS)

# --- 🛠️ 辅助函数：固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 保证cudnn可复现性（会降低一点速度，但在cpu上无影响）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            
            escaped_target = re.escape(target_name)
            pattern = escaped_target.replace(r'\ ', r'[._\s]+')
            match = re.search(pattern, input_path, re.IGNORECASE)
            
            if match:
                start_idx = match.start()
                end_idx = match.end()
                
                input_ids = [char_to_idx.get(c, 1) for c in input_path[:max_len]]
                labels = [0.0] * len(input_ids)
                
                limit = min(end_idx, max_len)
                for i in range(start_idx, limit):
                    labels[i] = 1.0
                
                pad_len = max_len - len(input_ids)
                self.samples.append((
                    torch.tensor(input_ids + [0] * pad_len), 
                    torch.tensor(labels + [0.0] * pad_len)
                ))
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"⚠️ 跳过了 {skipped_count} 条无法匹配标签的数据。")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- 🛠️ 辅助函数：验证集计算 ---
def validate_one_epoch(model, loader, criterion):
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for vx, vy in loader:
            pred = model(vx)
            loss = criterion(pred, vy)
            v_loss += loss.item()
    return v_loss / len(loader) if len(loader) > 0 else 0

# --- 训练逻辑 ---
def run_train():
    # 设置全局种子，保证后续 DataLoader shuffle 等行为一致
    set_seed(SEED)
    print(f"🔒 随机种子已固定为: {SEED}")

    if not os.path.exists(DATA_FILE): 
        print(f"❌ 找不到数据文件 {DATA_FILE}"); return
        
    with open(DATA_FILE, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
    
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
        print("ℹ️ 已加载现有词表。")
    else:
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
    if val_size < 1: train_size -= 1; val_size += 1
    
    # 使用固定种子的 Generator 进行数据集切分
    # 这样每次运行脚本，分到 train 和 val 的数据是完全固定的
    split_generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_generator)
    
    # DataLoader 的 shuffle=True 也会受到全局 torch.manual_seed 的影响
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FilmExtractor(len(char_to_idx))
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 初始化 best_val_loss 逻辑
    best_val_loss = float('inf')

    if os.path.exists(MODEL_PATH):
        print(f"🔄 检测到现有模型，加载权重以 LR={LR} 继续微调...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        
        # 在开始训练循环前，先计算一次当前模型的验证集 Loss
        print("📊 正在计算当前模型的初始验证集 Loss (基准线)...")
        initial_val_loss = validate_one_epoch(model, val_loader, criterion)
        best_val_loss = initial_val_loss # 将起点设为当前模型水平
        print(f"✅ 当前模型基准 Loss: {best_val_loss:.4f}")
    else:
        print("🆕 未检测到模型，将从头开始训练。")

    print(f"🚀 开始训练 | 样本数: {len(dataset)} | 训练集: {len(train_ds)} | 验证集: {len(val_ds)}")
    
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
            
            # 使用封装好的验证函数
            avg_val_loss = validate_one_epoch(model, val_loader, criterion)
            
            # 只有当 Loss 确实比之前的（包括刚加载进来的）更低时，才保存
            if avg_val_loss < best_val_loss:
                print(f" ✨ Loss 优化 ({best_val_loss:.4f} -> {avg_val_loss:.4f})，模型已更新。")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                print(f" ⏳ 验证集 Loss: {avg_val_loss:.4f} (未提升，最佳: {best_val_loss:.4f})")
                
    except KeyboardInterrupt: print("\n🛑 用户手动停止训练。")

# --- 预测逻辑 ---
def run_predict(path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("❌ 错误: 找不到模型或词表文件。请先运行训练。"); return

    with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
    model = FilmExtractor(len(char_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    input_ids = [char_to_idx.get(c, 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    
    with torch.no_grad():
        probs = model(torch.tensor([padded]))[0][:len(path)].numpy()

    if DEBUG_MODE:
        print(f"\n{'='*65}")
        print(f"{'索引':<4} | {'字符':<4} | {'分值':<15} | 状态")
        print("-" * 65)
        for i, p in enumerate(probs):
            status = "✅ [选中]" if p > THRESHOLD else "   [排除]"
            print(f"{i:<4} | {path[i]:<4} | {p:.10f} | {status}")
        print(f"{'='*65}\n")

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
    clean_result = raw_result.replace('.', ' ').replace('_', ' ')
    clean_result = re.sub(r'\s+', ' ', clean_result)
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
        run_train()
        