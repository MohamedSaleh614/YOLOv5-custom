import torch
import torch.nn as nn
import math

def YOLOv5Loss(pred, target, anchors, num_classes=80):
    device = pred.device
    dtype = pred.dtype
    B, A, S, H, W = pred.shape
    yv, xv = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device), indexing='ij')
    grid = torch.stack((xv, yv), dim=-1).to(dtype=dtype)
    #stride = 416.0 / S
    stride = float(640) / S
    stride = 640.0 / S if S == 80 else 416.0 / S if S == 52 else 640.0 / S

    anchors_grid = anchors.to(device=device, dtype=dtype) / stride
    anchor_w = anchors_grid[:, 0].view(1, A, 1, 1)
    anchor_h = anchors_grid[:, 1].view(1, A, 1, 1)

    px, py = pred[..., 0], pred[..., 1]
    pw, ph = pred[..., 2], pred[..., 3]
    bx = torch.sigmoid(px) + grid[..., 0]
    by = torch.sigmoid(py) + grid[..., 1]
    bw = anchor_w * torch.exp(pw)
    bh = anchor_h * torch.exp(ph)

    pred_x1 = bx - bw / 2
    pred_y1 = by - bh / 2
    pred_x2 = bx + bw / 2
    pred_y2 = by + bh / 2

    obj_mask = target[..., 4] > 0.5
    if obj_mask.sum() == 0:
        bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        return bce_obj(pred[..., 4], torch.zeros_like(pred[..., 4]))

    tx, ty = target[..., 0], target[..., 1]
    tw, th = target[..., 2], target[..., 3]
    gx = tx + grid[..., 0]
    gy = ty + grid[..., 1]
    gw = anchor_w * torch.exp(tw)
    gh = anchor_h * torch.exp(th)
    targ_x1 = gx - gw / 2
    targ_y1 = gy - gh / 2
    targ_x2 = gx + gw / 2
    targ_y2 = gy + gh / 2

    p_x1, p_y1, p_x2, p_y2 = pred_x1[obj_mask], pred_y1[obj_mask], pred_x2[obj_mask], pred_y2[obj_mask]
    t_x1, t_y1, t_x2, t_y2 = targ_x1[obj_mask], targ_y1[obj_mask], targ_x2[obj_mask], targ_y2[obj_mask]

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    p_w, p_h = p_x2 - p_x1, p_y2 - p_y1
    t_w, t_h = t_x2 - t_x1, t_y2 - t_y1
    union_area = (p_w * p_h) + (t_w * t_h) - inter_area + 1e-16
    iou = inter_area / union_area

    p_cx, p_cy = (p_x1 + p_x2)/2, (p_y1 + p_y2)/2
    t_cx, t_cy = (t_x1 + t_x2)/2, (t_y1 + t_y2)/2
    rho2 = (p_cx - t_cx)**2 + (p_cy - t_cy)**2

    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    c2 = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + 1e-16

    v = (4 / (math.pi ** 2)) * (torch.atan(t_w/t_h) - torch.atan(p_w/p_h)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-16)
    ciou = iou - (rho2 / c2) - alpha * v
    box_loss = (1 - ciou).mean()

    bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
    target_obj = torch.zeros_like(pred[..., 4])
    target_obj[obj_mask] = ciou.detach().clamp(0, 1).to(dtype)
    obj_loss = bce_obj(pred[..., 4], target_obj)

    if num_classes > 1:
        bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        t_cls = target[obj_mask][:, 5].long()
        t_cls_onehot = torch.zeros_like(pred[obj_mask][:, 5:5+num_classes])
        t_cls_onehot[torch.arange(t_cls.size(0)), t_cls] = 1.0
        class_loss = bce_cls(pred[obj_mask][:, 5:5+num_classes], t_cls_onehot)
    else:
        class_loss = torch.tensor(0.0, device=device)

    return box_loss * 0.05 + obj_loss * 1.0 + class_loss * 0.5