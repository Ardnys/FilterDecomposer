import os
import random
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from tqdm import tqdm
from fimage import FImage
from fimage.filters import (
    Brightness,
    Contrast,
    Exposure,
    Grayscale,
    Hue,
    Noise,
    Posterize,
    Saturation,
    Sepia,
    Sharpen,
    Vibrance,
)
from model import MultiOutputEfficientNet

IMG_SIZE = (260, 260)
FILTER_MAP = {
    "Contrast": (Contrast, (10, 75)),
    "Brightness": (Brightness, (5, 30)),
    "Saturation": (Saturation, (10, 100)),
    "Hue": (Hue, (0, 100)),
    "Exposure": (Exposure, (10, 30)),
    "Vibrance": (Vibrance, (20, 100)),
    "Sepia": (Sepia, (15, 100)),
    "Sharpen": (Sharpen, (10, 75)),
    "Noise": (Noise, (1, 30)),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiOutputEfficientNet(list(FILTER_MAP.keys()))
model.load_state_dict(torch.load("./models/fildec.pt", weights_only=True))
model.to(device)
model.eval()


def apply_random_filtering(img_path: Path, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    filters_to_apply = []
    for filter_name, (filter_class, (min_val, max_val)) in FILTER_MAP.items():
        if random.random() < 0.5:
            filters_to_apply.append(filter_class(random.randint(min_val, max_val)))

    image = FImage(img_path)
    for filter_obj in filters_to_apply:
        image.apply(filter_obj)

    if image.original_image.mode != "RGB":
        image = image.original_image.convert("RGB")

    image.save(output_dir / img_path.name)

    return output_dir / img_path.name


def apply_model_filtering(model, img_path: Path, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = model(input_tensor).squeeze(0).cpu().numpy()

    filters_to_apply = []
    for (filter_name, (filter_class, (min_val, max_val))), pred_value in zip(
        FILTER_MAP.items(), pred
    ):
        scaled_value = pred_value * (max_val - min_val) + min_val
        filters_to_apply.append(filter_class(scaled_value))

    image = FImage(img_path)
    for filter_obj in filters_to_apply:
        image.apply(filter_obj)

    image.save(output_dir / img_path.name)


def load_image(path: Path):
    return np.array(Image.open(path).convert("RGB")) / 255.0


def evaluate_with_ssim(original, filtered, predicted_filtered):
    ssim_value = ssim(filtered, predicted_filtered, channel_axis=-1, data_range=1.0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Original", "Filtered", f"Predicted Filtered\nSSIM: {ssim_value:.4f}"]
    images = [original, filtered, predicted_filtered]

    for i in range(3):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    fig.tight_layout()
    return fig, f"SSIM between filtered and predicted_filtered: {ssim_value:.4f}"


def pipeline(img_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = Path(img_path)
        temp_dir = Path(temp_dir)

        filtered_path = temp_dir / "filtered"
        predicted_path = temp_dir / "predicted_filtered"

        p = apply_random_filtering(img_path, filtered_path)
        apply_model_filtering(model, p, predicted_path)

        original = load_image(img_path)
        filtered = load_image(filtered_path / img_path.name)
        predicted_filtered = load_image(predicted_path / img_path.name)

        return evaluate_with_ssim(original, filtered, predicted_filtered)


gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[gr.Plot(label="Image Comparison"), gr.Textbox(label="SSIM Score")],
    title="Filter Decomposer Evaluation with Real Model",
).launch()
