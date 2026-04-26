import torch
import cv2
import numpy as np
import os
from torchvision.ops import nms

WEIGHTS_PATH = r"yolov5_best.pth"
IMAGE_PATH = r"/test.jpg"
IMG_SIZE = 640
CONF_THRES = 0.12
IOU_THRES = 0.45
NUM_CLASSES = 1

ANCHORS = torch.tensor([
    [5, 16],
    [10, 32],
    [14, 51],
    [20, 72],
    [23, 97],
    [30, 123],
    [34, 153],
    [40, 187],
    [70, 293]
], dtype=torch.float32)

def detect():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv5(num_classes=NUM_CLASSES).to(device)
    if not os.path.exists(WEIGHTS_PATH):
        print("Weights not found!")
        return
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    img0 = cv2.imread(IMAGE_PATH)
    if img0 is None:
        return
    h0, w0 = img0.shape[:2]

    img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img)

    output = torch.zeros((0, 6), device=device)
    strides = [8, 16, 32]
    anchors_device = ANCHORS.to(device).float()

    for i, (pred, stride) in enumerate(zip(preds, strides)):
        bs, _, ny, nx = pred.shape
        pred = pred.view(bs, 3, 5 + NUM_CLASSES, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
        grid = torch.stack((xv, yv), dim=2).view(1, 1, ny, nx, 2).float()

        current_anchors = anchors_device[i*3 : i*3 + 3].view(1, 3, 1, 1, 2)

        pred_xy = (torch.sigmoid(pred[..., :2]) + grid) * stride
        pred_wh = current_anchors * torch.exp(pred[..., 2:4])

        x1y1 = pred_xy - pred_wh / 2
        x2y2 = pred_xy + pred_wh / 2
        pred_box = torch.cat((x1y1, x2y2), dim=-1)

        conf = torch.sigmoid(pred[..., 4:5])
        cls_prob = torch.sigmoid(pred[..., 5:])

        detections = torch.cat((pred_box, conf, cls_prob), dim=-1).view(-1, 5 + NUM_CLASSES)
        output = torch.cat((output, detections), dim=0)

    mask = output[:, 4] > CONF_THRES
    output = output[mask]

    if output.shape[0] > 0:
        output[:, 5:] *= output[:, 4:5]
        scores, labels = output[:, 5:].max(1)
        keep = nms(output[:, :4], scores, IOU_THRES)
        final_dets = output[keep]
        final_scores = scores[keep]

        for i, box in enumerate(final_dets):
            x1, y1, x2, y2 = box[:4].cpu().numpy()
            x1 = int(x1 * w0 / IMG_SIZE)
            x2 = int(x2 * w0 / IMG_SIZE)
            y1 = int(y1 * h0 / IMG_SIZE)
            y2 = int(y2 * h0 / IMG_SIZE)
            conf = final_scores[i].item()
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img0, f"ID:{int(box[5])} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Detection", img0)
        cv2.waitKey(0)
    else:
        print("No objects detected.")

if __name__ == "__main__":
    detect()