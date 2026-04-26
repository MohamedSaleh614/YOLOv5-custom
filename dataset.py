import torch
from torchvision.ops import box_iou
from torchvision import io, transforms
import os
import math
from torch.utils.data import Dataset

def load_images(image_path, img_size=416, augment=False):
    image = io.read_image(image_path)
    flipped = False
    transforms_list = [
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=0., std=1.),
    ]
    if augment and torch.rand(1) < 0.5:
        transforms_list.insert(1, transforms.RandomHorizontalFlip(p=1.0))
        flipped = True
    transform = transforms.Compose(transforms_list)
    image = transform(image)
    return image.permute(1, 2, 0), flipped

def load_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return torch.empty((0,5), dtype=torch.float32)
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append([cls, x, y, w, h])
            except (ValueError, IndexError):
                continue
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0,5), dtype=torch.float32)

def build_targets(boxes, anchors, grid_sizes, num_classes, img_size):
    num_anchors = 3
    targets = [torch.zeros((num_anchors, s, s, 6), dtype=torch.float32) for s in grid_sizes]
    for box in boxes:
        cls, x, y, width, height = box
        cls = int(cls)
        for level_idx, grid_size in enumerate(grid_sizes):
            anchor_start = level_idx * 3
            anchor_end   = anchor_start + 3
            i, j = int(x * grid_size), int(y * grid_size)
            tx = x * grid_size - i
            ty = y * grid_size - j
            w_pixels = width * img_size
            h_pixels = height * img_size
            box_tensor = torch.tensor([[0, 0, w_pixels, h_pixels]], dtype=torch.float32)
            iou_list = []
            for anchor in anchors[anchor_start:anchor_end]:
                anchor_tensor = torch.tensor([[0, 0, anchor[0], anchor[1]]], dtype=torch.float32)
                iou = box_iou(box_tensor, anchor_tensor).item()
                iou_list.append(iou)
            best_anchor_idx = torch.tensor(iou_list).argmax().item()
            anchor_w, anchor_h = anchors[anchor_start + best_anchor_idx]
            tw = math.log(w_pixels / anchor_w + 1e-16)
            th = math.log(h_pixels / anchor_h + 1e-16)
            if 0 <= i < grid_size and 0 <= j < grid_size:
                targets[level_idx][best_anchor_idx, j, i, 0] = tx
                targets[level_idx][best_anchor_idx, j, i, 1] = ty
                targets[level_idx][best_anchor_idx, j, i, 2] = tw
                targets[level_idx][best_anchor_idx, j, i, 3] = th
                targets[level_idx][best_anchor_idx, j, i, 4] = 1.0
                targets[level_idx][best_anchor_idx, j, i, 5] = cls
    return targets

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, anchors, grid_sizes=[52, 26, 13], img_size=416, num_classes=20, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(img_dir))
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt").replace(".png", ".txt"))
        image, flipped = load_images(img_path, img_size=self.img_size, augment=self.augment)
        boxes = load_labels(label_path)

        if flipped and boxes.size(0) > 0:
            boxes[:, 1] = 1.0 - boxes[:, 1]

        targets = build_targets(
            boxes=boxes,
            anchors=self.anchors,
            grid_sizes=self.grid_sizes,
            num_classes=self.num_classes,
            img_size=self.img_size
        )
        return image, targets