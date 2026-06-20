import pandas as pd
import os
from PIL import Image

def generate_mock_images(csv_path):
    df = pd.read_csv(csv_path)
    
    count = 0
    for _, row in df.iterrows():
        raw_paths = row.get("image_paths", "")
        if pd.isna(raw_paths) or not isinstance(raw_paths, str):
            continue
            
        paths = [p.strip() for p in raw_paths.split(";") if p.strip()]
        for path in paths:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create and save a simple blank image if it doesn't exist
            if not os.path.exists(path):
                img = Image.new("RGB", (100, 100), color=(150, 150, 150))
                img.save(path)
                count += 1
                
    print(f"Successfully generated {count} mock images.")

if __name__ == "__main__":
    generate_mock_images("claims.csv")
