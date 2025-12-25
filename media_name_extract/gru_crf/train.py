import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchcrf import CRF
from tqdm import tqdm
import pickle
import os

# --- 核心配置 ---
NUM_THREADS = 4
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50
MAX_LEN = 300
MODEL_PATH = "movie_model_crf.pth"
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
            input_ids = [char_to_idx.get(c, 1) for c in input_path[:max_len]]
            
            # CRF 需要长整型标签，且取值 [0, 1]
            labels = [0] * max_len
            start_idx = input_path.find(target_name)
            if start_idx != -1:
                for i in range(start_idx, min(start_idx + len(target_name), max_len)):
                    labels[i] = 1
                
                pad_ids = input_ids + [0] * (max_len - len(input_ids))
                self.samples.append((torch.tensor(pad_ids), torch.tensor(labels)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class FilmExtractorCRF(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        # 映射到 2 个类别：0 (非片名), 1 (片名)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.crf = CRF(2, batch_first=True)

    def forward(self, x, tags=None):
        out, _ = self.gru(self.embedding(x))
        emissions = self.fc(out)
        if tags is not None:
            # 返回负对数似然作为损失值
            return -self.crf(emissions, tags, mask=(x != 0))
        else:
            # 预测模式：使用维特比算法解码最优路径
            return self.crf.decode(emissions, mask=(x != 0))

def train():
    if not os.path.exists(DATA_FILE): return
    with open(DATA_FILE, 'r', encoding='utf-8') as f: lines = f.readlines()
    
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
    else:
        all_chars = set("".join([l.split('#')[0] for l in lines if '#' in l]))
        char_to_idx = {c: i+2 for i, c in enumerate(sorted(list(all_chars)))}; char_to_idx['<PAD>'], char_to_idx['<UNK>'] = 0, 1
        with open(VOCAB_PATH, 'wb') as f: pickle.dump(char_to_idx, f)

    dataset = MovieDataset(lines, char_to_idx)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FilmExtractorCRF(len(char_to_idx))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    best_val_loss = float('inf')

    

    try:
        for epoch in range(EPOCHS):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}")
            for x, y in pbar:
                optimizer.zero_grad()
                loss = model(x, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.2f}")
            
            model.eval(); v_loss = 0
            with torch.no_grad():
                for vx, vy in val_loader: v_loss += model(vx, vy).item()
            avg_val_loss = v_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; torch.save(model.state_dict(), MODEL_PATH)
                print(f" ✨ Best Loss: {avg_val_loss:.2f}")
    except KeyboardInterrupt: print("\n🛑 Stopped.")

if __name__ == "__main__": train()