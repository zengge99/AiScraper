import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pickle
import os

# --- 核心配置 ---
NUM_THREADS = 4
BATCH_SIZE = 64
LR = 1e-4  # 学习率
EPOCHS = 50
MAX_LEN = 300  # 已修改为 300
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
DATA_FILE = "train_data.txt"

torch.set_num_threads(NUM_THREADS)

class MovieDataset(Dataset):
    def __init__(self, lines, char_to_idx, max_len=MAX_LEN):
        self.samples = []
        for line in lines:
            line = line.strip()
            if '#' not in line: continue
            input_path, target_name = line.rsplit('#', 1)
            # 转换 ID 序列，长度截断到 max_len
            input_ids = [char_to_idx.get(c, 1) for c in input_path[:max_len]]
            labels = [0.0] * len(input_ids)
            
            # 标注答案在路径中的位置
            start_idx = input_path.find(target_name)
            if start_idx != -1:
                for i in range(start_idx, min(start_idx + len(target_name), max_len)):
                    labels[i] = 1.0
                
                # 填充到固定长度 MAX_LEN
                pad_len = max_len - len(input_ids)
                self.samples.append((
                    torch.tensor(input_ids + [0] * pad_len), 
                    torch.tensor(labels + [0.0] * pad_len)
                ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

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

def train():
    if not os.path.exists(DATA_FILE): 
        print(f"❌ 找不到数据文件 {DATA_FILE}")
        return
        
    with open(DATA_FILE, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
    
    # 词表管理
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f: 
            char_to_idx = pickle.load(f)
        print("ℹ️ 已加载现有词表。")
    else:
        all_chars = set("".join([l.split('#')[0] for l in lines if '#' in l]))
        char_to_idx = {c: i+2 for i, c in enumerate(sorted(list(all_chars)))}
        char_to_idx['<PAD>'], char_to_idx['<UNK>'] = 0, 1
        with open(VOCAB_PATH, 'wb') as f: 
            pickle.dump(char_to_idx, f)
        print("🆕 已创建新词表。")

    dataset = MovieDataset(lines, char_to_idx)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREADS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

    model = FilmExtractor(len(char_to_idx))
    if os.path.exists(MODEL_PATH):
        print(f"🔄 检测到现有模型，加载权重以 LR={LR} 继续训练/微调...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    best_val_loss = float('inf') 

    print(f"🚀 开始训练 | 样本数: {len(dataset)} | 最大长度: {MAX_LEN}")
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
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止训练。")

if __name__ == "__main__":
    train()