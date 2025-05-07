import os
import csv
import random
from fimage import FImage
from fimage.filters import Sepia, Contrast, Brightness, Saturation, Hue

FILTERS = {
    'Contrast': Contrast,
    'Brightness': Brightness, 
    'Saturation': Saturation,
    'Hue': Hue,
    'Sepia': Sepia
}

def process_images(input_dir, output_dir, metadata_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(metadata_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id'] + list(FILTERS.keys()))
        writer.writeheader()
        
        image_files = [f for f in os.listdir(input_dir)]
        for img_file in image_files:            
            csv_row = {'Id': img_file, **{filter_name: 0 for filter_name in FILTERS}}

            filters_to_apply = []
            for filter_name, filter_class in FILTERS.items():
                if random.random() < 0.5:
                    value = random.randint(0, 85)
                    csv_row[filter_name] = value
                    filters_to_apply.append(filter_class(value))
            
            image = FImage(os.path.join(input_dir, img_file))
            for filter_obj in filters_to_apply:
                image.apply(filter_obj)
            
            image.save(os.path.join(output_dir, img_file))
            
            writer.writerow(csv_row)
            
            if image_files.index(img_file) % 100 == 0:
                print(f"Processed {image_files.index(img_file)}/{len(image_files)} images")
    
    print(f"Processing complete. Processed {len(image_files)} images.")
    print(f"Results saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

def main():
    input_directory = '../data/images'
    output_directory = '../results/images'
    csv_file = '../results/metadata.csv'
    
    process_images(input_directory, output_directory, csv_file)

if __name__ == "__main__":
    main()
