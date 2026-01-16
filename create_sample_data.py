"""
Create a synthetic Dogs vs Cats dataset for demonstration.
This generates placeholder images for testing the pipeline.
For real training, download the actual dataset from Kaggle.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def create_sample_image(label: str, index: int, size=(224, 224)):
    """Create a sample image with label text."""
    # Random color based on label
    if label == 'dog':
        base_color = (random.randint(150, 200), random.randint(100, 150), random.randint(50, 100))
    else:
        base_color = (random.randint(200, 255), random.randint(150, 200), random.randint(100, 150))
    
    img = Image.new('RGB', size, color=base_color)
    draw = ImageDraw.Draw(img)
    
    # Add some random shapes
    for _ in range(random.randint(3, 7)):
        shape_type = random.choice(['rectangle', 'ellipse'])
        x1, y1 = random.randint(0, size[0]-20), random.randint(0, size[1]-20)
        x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
        coords = [x1, y1, min(x2, size[0]), min(y2, size[1])]
        color = tuple(random.randint(0, 255) for _ in range(3))
        if shape_type == 'rectangle':
            draw.rectangle(coords, fill=color)
        else:
            draw.ellipse(coords, fill=color)
    
    # Add text
    text = f"{label.upper()} #{index}"
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    return img

def create_synthetic_dataset(data_dir: str = "./data", 
                            train_per_class: int = 1000,
                            val_per_class: int = 200, 
                            test_per_class: int = 200):
    """Create synthetic dataset with train/val/test splits."""
    data_path = Path(data_dir)
    
    splits = {
        'train': train_per_class,
        'val': val_per_class,
        'test': test_per_class
    }
    
    classes = ['dog', 'cat']
    
    print("Creating synthetic Dogs vs Cats dataset...")
    print("This is for demonstration only. Use real dataset for actual training.")
    
    total_images = 0
    for split, count in splits.items():
        for cls in classes:
            split_dir = data_path / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating {count} {cls} images for {split}...")
            for i in range(count):
                img = create_sample_image(cls, i)
                img_path = split_dir / f"{cls}.{i}.jpg"
                img.save(img_path, quality=85)
                total_images += 1
    
    print(f"\nSynthetic dataset created with {total_images} images:")
    print(f"Train: {train_per_class} dogs, {train_per_class} cats")
    print(f"Val: {val_per_class} dogs, {val_per_class} cats")
    print(f"Test: {test_per_class} dogs, {test_per_class} cats")
    print(f"\nDataset location: {data_path.absolute()}")

if __name__ == "__main__":
    # Create smaller dataset for faster testing
    create_synthetic_dataset(
        data_dir="./data",
        train_per_class=500,  # 500 per class = 1000 total train
        val_per_class=100,    # 100 per class = 200 total val
        test_per_class=100    # 100 per class = 200 total test
    )
