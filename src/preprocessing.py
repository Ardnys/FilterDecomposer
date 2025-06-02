import argparse
import os
from pathlib import Path
import shutil
import csv
import random
from tqdm import tqdm
from fimage import FImage
from fimage.filters import Contrast, Brightness, Saturation, Hue, Exposure, Grayscale, Sepia, Sharpen, Vibrance, Noise
from torchvision.datasets import CIFAR10

FILTERS = {
    'Contrast': (Contrast, (10, 75)),
    'Brightness': (Brightness, (5, 30)),
    'Saturation': (Saturation, (10, 100)),
    'Hue': (Hue, (0, 100)),
    'Exposure': (Exposure, (10, 30)),
    'Vibrance': (Vibrance, (20, 100)),
    'Pale': (Vibrance, (-100, -20)),
    'Sepia': (Sepia, (15, 100)),
    'Sharpen': (Sharpen, (10, 75)),
    'Noise': (Noise, (1, 30)),
    # 'Grayscale': (Grayscale, (0, 1)), on or off
}

# TODO: maybe using presets?

def prepare_cifar_images(input_dir: Path, num_images=1000):
    os.makedirs(input_dir, exist_ok=True)
    temp_dir = Path('./temp_cifar')

    cifar = CIFAR10(root=temp_dir, download=True)

    for idx in range(min(num_images, len(cifar))):
        image, _ = cifar[idx]
        image.save(input_dir / f"cifar_{idx}.png")

    shutil.rmtree(temp_dir)

def get_img_paths(input_dir: Path, recursive=False):
    if recursive:
        return list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg"))
    else:
        return list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))

def apply_filter(img_path: Path, input_dir: Path, output_dir: Path, writer: csv.DictWriter, idx):
    rel_path = img_path.relative_to(input_dir)
    rel_path = rel_path.with_name(f"{rel_path.stem}_v{idx}{rel_path.suffix}")
    output_path = output_dir / rel_path
    # recursively create subfolders
    os.makedirs(output_path.parent, exist_ok=True)

    csv_row = {'Id': str(rel_path), **{filter_name: 0 for filter_name in FILTERS}}

    filters_to_apply = []
    for filter_name, (filter_class, (min_val, max_val)) in FILTERS.items():
        if random.random() < 0.5:
            value = random.randint(min_val, max_val)
            csv_row[filter_name] = value
            filters_to_apply.append(filter_class(value))


    image = FImage(img_path)

    for filter_obj in filters_to_apply:
        image.apply(filter_obj)

    if image.original_image.mode != "RGB":
        image = image.original_image.convert("RGB")

    image.save(output_path)
    writer.writerow(csv_row)

def process_images(input_dir: Path, output_dir: Path, metadata_path: Path, recursive=False, filter_per_image=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id'] + list(FILTERS.keys()))
        writer.writeheader()
        
        image_paths = get_img_paths(input_dir, recursive)

        progress_bar = tqdm(image_paths, desc="Applying filters", unit="image")

        weird_images = 0
        for img_path in progress_bar:    
            for i in range(1, filter_per_image+1):
                try:
                    apply_filter(img_path, input_dir, output_dir, writer, i)
                except ValueError as e:
                    weird_images += 1
    
    print(f"Processing complete. Processed {len(image_paths) - weird_images} images.")
    print(f"{weird_images} images could not be processed for some odd reason.")
    print(f"Results saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Apply filters to image datasets")
    parser.add_argument("-i", "--input-dir", type=Path, help="Path to input image directory")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path('./results/images'), help="Path to directory to save processed images")
    parser.add_argument('--metadata', type=Path, default=Path('./results/metadata.csv'), help="Path to CSV file to store filter metadata")
    parser.add_argument("--dataset", type=str, choices=["cifar10"], help="Use torchvision datasets for input images")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to use from dataset if --dataset is provided")
    parser.add_argument("-f", "--filters-per-image", type=int, default=1, help="Number of filter combinations to apply per image")
    parser.add_argument("-r", '--recursive', action="store_true", help="Recursively search for images in subdirectories of input directory")

    args = parser.parse_args()

    if not args.input_dir and not args.dataset:
        raise ValueError("You must provide either --input-dir or --dataset")

    input_dir = args.input_dir or Path("../data/images")

    if args.dataset:
        print(f"Downloading dataset: {args.dataset}")
        prepare_cifar_images(input_dir, num_images=args.num_images)

    process_images(input_dir, args.output_dir, args.metadata, args.recursive, args.filters_per_image)

if __name__ == "__main__":
    main()
