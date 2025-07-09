import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets.text_dataset import TextEmotionDataset
from utils.label_map import label_map, id2label
from models.text_encoder import TextEncoder
from sklearn.metrics import accuracy_score
import os

# --------- 設定 ---------
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- 資料讀取 ---------
print("[INFO] Loading dataset...")
train_dataset = TextEmotionDataset("data/train_sent_emo.csv", label_map, max_len=MAX_LEN)
dev_dataset = TextEmotionDataset("data/dev_sent_emo.csv", label_map, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# --------- 模型建立 ---------
print("[INFO] Building model...")
model = TextEncoder(num_labels=len(label_map))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)

# --------- 訓練迴圈 ---------
def train_epoch(model, loader):
    model.train()
    total_loss, total_correct = 0, 0
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()

    acc = total_correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

# --------- 驗證迴圈 ---------
def eval_model(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return acc

# --------- 主訓練流程 ---------
print("[INFO] Starting training...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc = eval_model(model, dev_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Acc: {val_acc:.4f}")

# --------- 儲存模型 ---------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/text_model.pt")
print("[INFO] Model saved to checkpoints/text_model.pt")
