import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn import YOLOv5
from dataset import YOLODataset
from loss import YOLOv5Loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 640
NUM_CLASSES = 1

ANCHORS = torch.tensor([
    [5, 16],
    [9, 32],
    [13, 50],
    [19, 72],
    [22, 97],
    [29, 122],
    [34, 153],
    [40, 187],
    [69, 292]
], dtype=torch.float32)

GRID_SIZES = [IMG_SIZE // 8, IMG_SIZE // 16, IMG_SIZE // 32]

def collate_fn(batch):
    images = torch.stack([b[0].permute(2,0,1) for b in batch])
    targets_small   = torch.stack([b[1][0] for b in batch])
    targets_medium  = torch.stack([b[1][1] for b in batch])
    targets_large   = torch.stack([b[1][2] for b in batch])
    return images, [targets_small, targets_medium, targets_large]

if __name__ == "__main__":
    model = YOLOv5(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    dataset = YOLODataset(
        r"\images",
        r"\labels",
        ANCHORS,
        GRID_SIZES,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        for images, targets in loop:
            images = images.to(DEVICE)
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss = 0.0
                for i, (pred, target) in enumerate(zip(preds, targets)):
                    B, C, H, W = pred.shape
                    pred = pred.view(B, 3, 5 + NUM_CLASSES, H, W).permute(0, 1, 3, 4, 2)
                    anchors_slice = ANCHORS[i*3:(i+1)*3].to(DEVICE)
                    loss += YOLOv5Loss(pred, target.to(DEVICE), anchors_slice, num_classes=NUM_CLASSES)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

    torch.save(model.state_dict(), "yolov5_best.pth")