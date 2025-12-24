import os
import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 从PyTorch导入AdamW
from model import MediaNameExtractor
from dataset import SimpleMediaDataset

# 忽略冗余警告
warnings.filterwarnings("ignore")

# 全局配置（VPS CPU适配）
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
MAX_PATH_LEN = 128  # 路径最长128字符
MAX_NAME_LEN = 32   # 名称最长32字符
EPOCHS = 15
LR = 3e-5
SAVE_PATH = "best_media_model.pt"

def train(args):
    """训练模型（只需路径#名称格式数据）"""
    # 1. 初始化分词器和数据集（删除use_fast=False）
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    train_dataset = SimpleMediaDataset(args.train_data, tokenizer, MAX_PATH_LEN, MAX_NAME_LEN)
    dev_dataset = SimpleMediaDataset(args.dev_data, tokenizer, MAX_PATH_LEN, MAX_NAME_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = MediaNameExtractor().to(DEVICE)

    # 3. 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        eps=1e-8,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 4. 训练循环
    best_dev_loss = float("inf")
    print(f"===== 开始训练（CPU模式）=====")
    print(f"训练数据量：{len(train_dataset)} | 验证数据量：{len(dev_dataset)}")
    print(f"批次大小：{BATCH_SIZE} | 路径最大长度：{MAX_PATH_LEN}")

    for epoch in range(EPOCHS):
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

        # 验证阶段
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

        # 保存最优模型
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[Epoch {epoch+1}] 验证损失下降，保存模型到 {SAVE_PATH}")

        print(f"[Epoch {epoch+1}] 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_dev_loss:.4f}")

    print(f"===== 训练完成！模型已保存为 {SAVE_PATH} =====")

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
    parser = argparse.ArgumentParser(description="影视名称提取模型（简化标注版）")
    subparsers = parser.add_subparsers(dest="command", help="子命令：train / infer")

    # 训练子命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--train_data", type=str, default="train_data.txt", help="训练数据路径（路径#名称格式）")
    train_parser.add_argument("--dev_data", type=str, default="dev_data.txt", help="验证数据路径")

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