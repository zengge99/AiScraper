import torch
import torch.nn as nn
import pickle
import sys
import os
from torchcrf import CRF

# --- 核心配置 ---
MAX_LEN = 300
MODEL_PATH = "movie_model_crf.pth"
VOCAB_PATH = "vocab.pkl"

class FilmExtractorCRF(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.crf = CRF(2, batch_first=True)

    def forward(self, x, tags=None):
        out, _ = self.gru(self.embedding(x))
        emissions = self.fc(out)
        if tags is not None:
            return -self.crf(emissions, tags, mask=(x != 0))
        else:
            return self.crf.decode(emissions, mask=(x != 0)), emissions

def predict():
    if len(sys.argv) < 2: return
    path = sys.argv[1].strip()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(curr_dir, VOCAB_PATH), 'rb') as f:
        char_to_idx = pickle.load(f)
    
    model = FilmExtractorCRF(len(char_to_idx))
    model.load_state_dict(torch.load(os.path.join(curr_dir, MODEL_PATH), map_location='cpu'))
    model.eval()

    input_ids = [char_to_idx.get(c, 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    x_tensor = torch.tensor([padded])

    

    with torch.no_grad():
        # best_path: [[0, 0, 1, 1, 0...]]
        best_path, emissions = model(x_tensor)
        best_path = best_path[0]
        # 计算伪概率用于调试打印 (Softmax 处理发射得分)
        probs = torch.softmax(emissions[0], dim=-1)[:, 1].numpy()

    print(f"\n{'='*65}")
    print(f"{'索引':<4} | {'字符':<4} | {'片名概率 (10位)':<15} | {'CRF 决策'}")
    print("-" * 65)
    
    res_list = []
    for i in range(len(path)):
        is_selected = best_path[i] == 1
        status = "✅ [选中]" if is_selected else "   [排除]"
        print(f"{i:<4} | {path[i]:<4} | {probs[i]:.10f} | {status}")
        if is_selected:
            res_list.append(path[i])
            
    print(f"{'='*65}\n")
    print(f"最终提取结果: {''.join(res_list).strip('/()# ')}")

if __name__ == "__main__":
    predict()