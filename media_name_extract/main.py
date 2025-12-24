import os
import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import MediaNameExtractor
from dataset import SimpleMediaDataset

warnings.filterwarnings("ignore")

# 全局配置
DEVICE = torch.device("cpu")
BATCH_SIZE = 4
MAX_PATH_LEN = 128
MAX_NAME_LEN = 32
LR = 5e-5  # 提高学习率，加速BERT微调
SAVE_PATH = "best_media_model.pt"
LOSS_RECORD_PATH = "best_loss.txt"

def load_best_loss():
    if os.path.exists(LOSS_RECORD_PATH):
        try:
            with open(LOSS_RECORD_PATH, "r", encoding="utf-8") as f:
                loss = float(f.read().strip())
            return loss
        except:
            return float("inf")
    return float("inf")

def save_best_loss(loss):
    with open(LOSS_RECORD_PATH, "w", encoding="utf-8") as f:
        f.write(f"{loss:.6f}")

def train(args):
    # 初始化分词器和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = MediaNameExtractor().to(DEVICE)
    
    # 加载数据
    train_dataset = SimpleMediaDataset(args.train_data, tokenizer, MAX_PATH_LEN, MAX_NAME_LEN)
    dev_dataset = SimpleMediaDataset(args.dev_data, tokenizer, MAX_PATH_LEN, MAX_NAME_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载最优模型（如有）
    best_dev_loss = load_best_loss()
    model_loaded = False
    if os.path.exists(SAVE_PATH):
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
            model.eval()
            init_dev_loss = 0.0
            with torch.no_grad():
                for batch in dev_loader:
                    path_ids = batch["path_ids"].to(DEVICE)
                    path_mask = batch["path_mask"].to(DEVICE)
                    name_ids = batch["name_ids"].to(DEVICE)
                    name_mask = batch["name_mask"].to(DEVICE)
                    outputs = model(path_ids, path_mask, name_ids, name_mask)
                    init_dev_loss += outputs["loss"].item()
            best_dev_loss = init_dev_loss / len(dev_loader)
            model_loaded = True
            print(f"✅ 加载最优模型，历史最优验证损失：{best_dev_loss:.4f}")
        except Exception as e:
            print(f"⚠️  加载模型失败，从头训练：{e}")
            best_dev_loss = float("inf")

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    # 训练循环
    print(f"\n===== 开始训练 =====")
    print(f"训练轮数：{args.epochs} | 批次大小：{BATCH_SIZE} | 学习率：{LR}")
    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            path_ids = batch["path_ids"].to(DEVICE)
            path_mask = batch["path_mask"].to(DEVICE)
            name_ids = batch["name_ids"].to(DEVICE)
            name_mask = batch["name_mask"].to(DEVICE)

            # 前向+反向传播
            outputs = model(path_ids, path_mask, name_ids, name_mask)
            loss = outputs["loss"]
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss_total += loss.item()

            # 打印批次损失
            if (batch_idx + 1) % 5 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Batch {batch_idx+1} | 批次损失：{loss.item():.4f}")

        # 验证阶段
        model.eval()
        dev_loss_total = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                path_ids = batch["path_ids"].to(DEVICE)
                path_mask = batch["path_mask"].to(DEVICE)
                name_ids = batch["name_ids"].to(DEVICE)
                name_mask = batch["name_mask"].to(DEVICE)
                outputs = model(path_ids, path_mask, name_ids, name_mask)
                dev_loss_total += outputs["loss"].item()

        # 计算平均损失
        avg_train_loss = train_loss_total / len(train_loader)
        avg_dev_loss = dev_loss_total / len(dev_loader)

        # 保存最优模型
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), SAVE_PATH)
            save_best_loss(best_dev_loss)
            print(f"[Epoch {epoch+1}] 🎉 验证损失下降：{best_dev_loss:.4f}，保存模型")
        else:
            print(f"[Epoch {epoch+1}] ❌ 验证损失未下降：当前={avg_dev_loss:.4f} | 最优={best_dev_loss:.4f}")

        print(f"[Epoch {epoch+1}] 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_dev_loss:.4f}\n")

    print(f"===== 训练完成！最优模型已保存至 {SAVE_PATH} =====")

def infer(args):
    # 加载模型
    if not os.path.exists(SAVE_PATH):
        print(f"错误：未找到模型文件 {SAVE_PATH}，请先训练！")
        return

    model = MediaNameExtractor().to(DEVICE)
    try:
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 提取名称
    model.extract_name(args.path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="影视名称提取模型（解决分数集中0.5问题）")
    subparsers = parser.add_subparsers(dest="command", help="子命令：train / infer")

    # 训练参数
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--train_data", type=str, default="train_data.txt", help="训练数据路径")
    train_parser.add_argument("--dev_data", type=str, default="dev_data.txt", help="验证数据路径")
    train_parser.add_argument("--epochs", type=int, default=20, help="训练轮数（建议20轮）")

    # 推理参数
    infer_parser = subparsers.add_parser("infer", help="提取影视名称")
    infer_parser.add_argument("--path", type=str, required=True, help="待提取的文件路径")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()