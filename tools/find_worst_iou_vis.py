import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import re
import csv
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model


def safe_name(name):
    name = str(name)
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return name


def load_checkpoint(model, weight_path, device):
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"加载权重完成: {weight_path}")
    print(f"missing_keys: {len(missing)}, unexpected_keys: {len(unexpected)}")

    if len(missing) > 0:
        print("前 10 个 missing_keys:")
        for k in missing[:10]:
            print("  ", k)

    if len(unexpected) > 0:
        print("前 10 个 unexpected_keys:")
        for k in unexpected[:10]:
            print("  ", k)


def get_logits(output):
    if isinstance(output, dict):
        if "final_logits" in output:
            return output["final_logits"]
        if "logits" in output:
            return output["logits"]
        raise RuntimeError("模型输出是 dict，但没有 final_logits 或 logits 字段。")

    if isinstance(output, (list, tuple)):
        return output[0]

    return output


def tensor_to_rgb(img_tensor):
    """
    把 dataloader 里的图像 tensor 转成 RGB numpy，便于画图。
    兼容 0-1、0-255、ImageNet Normalize 后的输入。
    """
    img = img_tensor.detach().cpu().float()

    if img.dim() == 4:
        img = img[0]

    img = img.numpy()

    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)

    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # 处理 ImageNet Normalize 的情况
    if img.min() < -0.1 or img.max() > 1.5:
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img = img * std + mean

    # 处理 0-255 的情况
    if img.max() > 2.0:
        img = img / 255.0

    img = np.clip(img, 0.0, 1.0)
    return img


def mask_to_rgb(mask_bool):
    mask = mask_bool.astype(np.uint8) * 255
    rgb = np.stack([mask, mask, mask], axis=-1)
    return rgb


def make_error_map(gt_bool, pred_bool):
    """
    绿色 TP
    红色 FP
    蓝色 FN
    黑色 TN
    """
    tp = pred_bool & gt_bool
    fp = pred_bool & (~gt_bool)
    fn = (~pred_bool) & gt_bool

    h, w = gt_bool.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[tp] = np.array([0, 255, 0], dtype=np.uint8)      # TP green
    vis[fp] = np.array([255, 0, 0], dtype=np.uint8)      # FP red
    vis[fn] = np.array([0, 80, 255], dtype=np.uint8)     # FN blue

    return vis, tp, fp, fn


def compute_metrics(gt_bool, pred_bool):
    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
    fn = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def save_four_panel(
    save_path,
    image_rgb,
    gt_rgb,
    pred_rgb,
    error_rgb,
    title_info,
):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=300)

    panels = [
        (image_rgb, "Image"),
        (gt_rgb, "Ground Truth"),
        (pred_rgb, "Prediction"),
        (error_rgb, "Error Map"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # 误差图图例
    legend_handles = [
        Patch(facecolor=(0 / 255, 255 / 255, 0 / 255), edgecolor="black",
              label="TP: 正确预测为道路"),
        Patch(facecolor=(255 / 255, 0 / 255, 0 / 255), edgecolor="black",
              label="FP: 背景误检为道路"),
        Patch(facecolor=(0 / 255, 80 / 255, 255 / 255), edgecolor="black",
              label="FN: 真实道路被漏检"),
    ]

    axes[3].legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=8,
        frameon=True,
        ncol=1,
    )

    # 在误差图标题上也直接标注颜色含义
    axes[3].set_title(
        "Error Map\nGreen=TP, Red=FP, Blue=FN",
        fontsize=10,
    )

    fig.suptitle(title_info, fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="config.json 路径")
    parser.add_argument("-w", "--weight", type=str, required=True, help="best_model.pth 路径")
    parser.add_argument("--out_dir", type=str, default="vis_worst_iou", help="可视化输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="预测阈值")
    parser.add_argument("--topk", type=int, default=3, help="保存 IoU 最低的前 k 张")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset"]["name"]
    root_path = config["dataset"]["root_path"]
    img_size = config["dataset"].get("input_size", 1024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "DRIVE":
        test_dataset = DRIVEDataset(
            root_path,
            dataset_name,
            mode="test",
            img_size=img_size,
        )
    else:
        test_dataset = RoadDataset(
            root_path,
            dataset_name,
            mode="test",
            img_size=img_size,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = get_model(config["model"], img_size=img_size).to(device)
    load_checkpoint(model, args.weight, device)
    model.eval()

    all_results = []

    print("开始逐图计算 IoU...")

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            if len(batch_data) == 3:
                imgs, masks, names = batch_data
                name = names[0]
            else:
                imgs, masks = batch_data
                name = f"sample_{idx:03d}"

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            outputs = model(imgs)
            logits = get_logits(outputs)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            probs = torch.sigmoid(logits)
            preds = probs > args.threshold
            gt = masks > 0.5

            pred_bool = preds[0, 0].detach().cpu().numpy().astype(bool)

            if gt.dim() == 4:
                gt_bool = gt[0, 0].detach().cpu().numpy().astype(bool)
            else:
                gt_bool = gt[0].detach().cpu().numpy().astype(bool)

            metrics = compute_metrics(gt_bool, pred_bool)

            image_rgb = tensor_to_rgb(imgs[0])
            gt_rgb = mask_to_rgb(gt_bool)
            pred_rgb = mask_to_rgb(pred_bool)
            error_rgb, tp_map, fp_map, fn_map = make_error_map(gt_bool, pred_bool)

            item = {
                "idx": idx,
                "name": safe_name(name),
                "raw_name": str(name),
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "image_rgb": image_rgb,
                "gt_rgb": gt_rgb,
                "pred_rgb": pred_rgb,
                "error_rgb": error_rgb,
            }

            all_results.append(item)

            print(
                f"[{idx + 1:02d}/{len(test_dataset):02d}] "
                f"{item['name']} | "
                f"IoU={metrics['iou']:.4f}, "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"FP={metrics['fp']}, FN={metrics['fn']}"
            )

    all_results_sorted = sorted(all_results, key=lambda x: x["iou"])
    worst_items = all_results_sorted[:args.topk]

    csv_path = os.path.join(args.out_dir, "per_image_metrics_sorted.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "idx",
                "name",
                "iou",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
            ]
        )

        for rank, item in enumerate(all_results_sorted, start=1):
            writer.writerow(
                [
                    rank,
                    item["idx"],
                    item["raw_name"],
                    f"{item['iou']:.6f}",
                    f"{item['precision']:.6f}",
                    f"{item['recall']:.6f}",
                    f"{item['f1']:.6f}",
                    item["tp"],
                    item["fp"],
                    item["fn"],
                ]
            )

    print("\nIoU 最低的样本：")

    for rank, item in enumerate(worst_items, start=1):
        title_info = (
            f"Rank {rank} | {item['name']} | "
            f"IoU {item['iou']:.4f} | "
            f"P {item['precision']:.4f} | "
            f"R {item['recall']:.4f} | "
            f"F1 {item['f1']:.4f}"
        )

        save_name = (
            f"rank{rank:02d}_"
            f"iou{item['iou']:.4f}_"
            f"{item['name']}.png"
        )

        save_path = os.path.join(args.out_dir, save_name)

        save_four_panel(
            save_path=save_path,
            image_rgb=item["image_rgb"],
            gt_rgb=item["gt_rgb"],
            pred_rgb=item["pred_rgb"],
            error_rgb=item["error_rgb"],
            title_info=title_info,
        )

        print(
            f"Rank {rank}: {item['name']} | "
            f"IoU={item['iou']:.4f}, "
            f"P={item['precision']:.4f}, "
            f"R={item['recall']:.4f}, "
            f"F1={item['f1']:.4f} | "
            f"saved: {save_path}"
        )

    print(f"\n逐图指标已保存: {csv_path}")
    print(f"可视化结果目录: {args.out_dir}")


if __name__ == "__main__":
    main()