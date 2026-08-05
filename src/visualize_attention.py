import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders
from model import AttentionCNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_attention_cnn.pt")


def get_spatial_attention_maps(model, images):
    """استخراج spatial attention maps از هر لایه"""
    device = next(model.parameters()).device
    images = images.to(device)

    attention_maps = []

    x = model.conv1(images)
    ch_avg = x.mean(dim=1, keepdim=True)
    ch_max = x.max(dim=1, keepdim=True)[0]
    cat = torch.cat([ch_avg, ch_max], dim=1)
    att1 = model.att1.spatial_att.conv(cat)
    attention_maps.append(att1.detach().cpu())
    x = model.att1(x)

    x = model.conv2(x)
    ch_avg = x.mean(dim=1, keepdim=True)
    ch_max = x.max(dim=1, keepdim=True)[0]
    cat = torch.cat([ch_avg, ch_max], dim=1)
    att2 = model.att2.spatial_att.conv(cat)
    att2_upsampled = F.interpolate(att2, size=(32, 32), mode="bilinear", align_corners=False)
    attention_maps.append(att2_upsampled.detach().cpu())
    x = model.att2(x)

    x = model.conv3(x)
    ch_avg = x.mean(dim=1, keepdim=True)
    ch_max = x.max(dim=1, keepdim=True)[0]
    cat = torch.cat([ch_avg, ch_max], dim=1)
    att3 = model.att3.spatial_att.conv(cat)
    att3_upsampled = F.interpolate(att3, size=(32, 32), mode="bilinear", align_corners=False)
    attention_maps.append(att3_upsampled.detach().cpu())

    return attention_maps


def denormalize(img_tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(batch_size=8)

    model = AttentionCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    images, labels = next(iter(test_loader))
    attention_maps = get_spatial_attention_maps(model, images)

    n_samples = 4
    fig, axes = plt.subplots(n_samples, 5, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image", fontsize=9)
        axes[i, 0].axis("off")

        for layer_idx, att_map in enumerate(attention_maps):
            att = att_map[i, 0].numpy()
            att = (att - att.min()) / (att.max() - att.min() + 1e-8)
            axes[i, layer_idx + 1].imshow(att, cmap="hot")
            axes[i, layer_idx + 1].set_title(f"Layer {layer_idx + 1} Attention", fontsize=9)
            axes[i, layer_idx + 1].axis("off")

    plt.suptitle("Spatial Attention Maps (What does the model look at?)")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "attention_visualization.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved attention visualization to {out_path}")


if __name__ == "__main__":
    visualize()
