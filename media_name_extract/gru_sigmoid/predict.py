import torch
import torch.nn as nn
import pickle
import sys
import os

# --- 调试与提取配置 ---
DEBUG_MODE = True    # 开启后显示全路径所有字符得分
THRESHOLD = 0.2      # 核心判定阈值
SMOOTH_VAL = 0.05    # 辅助判定阈值（用于救回中间字符）
MAX_LEN = 300        # 已修改为 300
# --------------------

torch.set_num_threads(1)

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
        x = self.embedding(x)
        out, _ = self.gru(x)
        return self.fc(out).squeeze(-1)

def predict():
    if len(sys.argv) < 2:
        print("使用方法: python predict.py \"你的路径\"")
        return
    
    path = sys.argv[1].strip()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, 'movie_model.pth')
    vocab_path = os.path.join(curr_dir, 'vocab.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print("❌ 错误: 找不到模型或词表文件。")
        return

    # 1. 加载资源
    with open(vocab_path, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    model = FilmExtractor(len(char_to_idx))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. 预处理
    # 截取前 300 个字符
    input_ids = [char_to_idx.get(c, 1) for c in path[:MAX_LEN]]
    # 填充到 300 长度
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    
    # 3. 推理
    with torch.no_grad():
        # 获取模型评分结果，取实际路径长度对应的部分
        probs = model(torch.tensor([padded]))[0][:len(path)].numpy()

    # 4. 详细调试输出
    if DEBUG_MODE:
        print(f"\n{'='*65}")
        print(f"{'索引':<4} | {'字符':<4} | {'分值 (10位小数)':<15} | 状态")
        print("-" * 65)
        for i, p in enumerate(probs):
            status = "✅ [选中]" if p > THRESHOLD else "   [排除]"
            print(f"{i:<4} | {path[i]:<4} | {p:.10f} | {status}")
        print(f"{'='*65}\n")

    # 5. 增强型提取逻辑 (救回中间断开的字符)
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
    
    final_res = "".join(res_list).strip("/()# ")
    
    if DEBUG_MODE:
        print(f"最终提取结果: {final_res}")
    else:
        print(final_res)

if __name__ == "__main__":
    predict()