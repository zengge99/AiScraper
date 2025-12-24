import os
import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import MediaNameExtractor
from dataset import SimpleMediaDataset

# 忽略冗余警告
warnings.filterwarnings("ignore")

# 全局配置（VPS CPU适配）
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
MAX_PATH_LEN = 128
MAX_NAME_LEN = 32
LR = 3e-5
SAVE_PATH = "best_media_model.pt"  # 最优模型保存路径
LOSS_RECORD_PATH = "best_loss.txt" # 记录最优验证损失的文件

def load_best_loss():
    """加载历史最优验证损失"""
    if os.path.exists(LOSS_RECORD_PATH):
        try:
            with open(LOSS_RECORD_PATH, "r", encoding="utf-8") as f:
                loss = float(f.read().strip())
            return loss
        except:
            return float("inf")
    return float("inf")

def save_best_loss(loss):
    """保存最新最优验证损失"""
    with open(LOSS_RECORD_PATH, "w", encoding="utf-8") as f:
        f.write(f"{loss:.6f}")

def train(args):
    """训练模型：自动加载最优模型+读取最新数据"""
    # 1. 初始化分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    # 2. 读取最新的训练/验证数据（每次训练都会重新读取，自动用新数据）
    print(f"📖 正在读取最新数据：训练集={args.train_data} | 验证集={args.dev_data}")
    train_dataset = SimpleMediaDataset(args.train_data, tokenizer, MAX_PATH_LEN, MAX_NAME_LEN)
    dev_dataset = SimpleMediaDataset(args.dev_data, tokenizer, MAX_PATH_LEN, MAX_NAME_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型（自动检测并加载最优模型）
    model = MediaNameExtractor().to(DEVICE)
    best_dev_loss = load_best_loss()  # 加载历史最优损失
    model_loaded = False
    
    if os.path.exists(SAVE_PATH):
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
            model_loaded = True
            print(f"✅ 自动加载最优模型：{SAVE_PATH}，历史最优验证损失：{best_dev_loss:.4f}")
        except Exception as e:
            print(f"⚠️  加载模型失败，将从头训练：{str(e)}")
            best_dev_loss = float("inf")
    else:
        print("🚀 未找到最优模型，将从头训练")

    # 4. 优化器配置
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        eps=1e-8,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 5. 训练循环（使用最新数据继续训练）
    print(f"\n===== 开始训练（CPU模式）=====")
    print(f"训练数据量：{len(train_dataset)} | 验证数据量：{len(dev_dataset)}")
    print(f"训练轮数：{args.epochs} | 已加载模型：{model_loaded}")

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            path_ids = batch["path_ids"].to(DEVICE)
            path_mask = batch["path_mask"].to(DEVICE)
            name_ids = batch["name_ids"].to(DEVICE)
            name_mask = batch["name_mask"].to(DEVICE)

            # 前向传播
            outputs = model(path_ids, path_mask, name_ids, name_mask)
            loss = outputs["loss"]
            train_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 验证阶段（用最新验证数据评估）
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                path_ids = batch["path_ids"].to(DEVICE)
                path_mask = batch["path_mask"].to(DEVICE)
                name_ids = batch["name_ids"].to(DEVICE)
                name_mask = batch["name_mask"].to(DEVICE)
                outputs = model(path_ids, path_mask, name_ids, name_mask)
                dev_loss += outputs["loss"].item()

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_dev_loss = dev_loss / len(dev_loader)

        # 保存更优的模型和损失（只有验证损失更低时才更新）
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), SAVE_PATH)
            save_best_loss(best_dev_loss)
            print(f"[Epoch {epoch+1}] 🎉 验证损失下降 ({best_dev_loss:.4f})，更新最优模型")
        else:
            print(f"[Epoch {epoch+1}] ❌ 验证损失未下降 (当前：{avg_dev_loss:.4f} | 最优：{best_dev_loss:.4f})")

        print(f"[Epoch {epoch+1}] 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_dev_loss:.4f}")

    print(f"\n===== 训练完成！最优模型：{SAVE_PATH} =====")

def infer(args):
    """推理：从路径提取名称"""
    # 1. 检查模型
    if not os.path.exists(SAVE_PATH):
        print(f"错误：未找到模型 {SAVE_PATH}，请先训练！")
        return

    # 2. 加载模型
    model = MediaNameExtractor().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    # 3. 提取名称
    path = args.path
    print(f"原始路径：{path}")
    name = model.extract_name(path)
    print(f"提取的影视名称：{name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="影视名称提取模型（自动续训+读取最新数据）")
    subparsers = parser.add_subparsers(dest="command", help="子命令：train / infer")

    # 训练子命令（极简使用，默认参数即可）
    train_parser = subparsers.add_parser("train", help="训练模型（自动加载最优模型+最新数据）")
    train_parser.add_argument("--train_data", type=str, default="train_data.txt", help="训练数据路径（默认train_data.txt）")
    train_parser.add_argument("--dev_data", type=str, default="dev_data.txt", help="验证数据路径（默认dev_data.txt）")
    train_parser.add_argument("--epochs", type=int, default=5, help="每次训练的轮数（默认5轮，建议小轮数多次训）")

    # 推理子命令
    infer_parser = subparsers.add_parser("infer", help="提取影视名称")
    infer_parser.add_argument("--path", type=str, required=True, help="原始文件路径")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()