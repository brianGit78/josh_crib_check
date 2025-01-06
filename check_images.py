import os
from PIL import Image
import creds
from file_sync import FileManager

file_manager = FileManager(creds.model_name)

#root_folder = '/path/to/your/images'
root_folder = os.path.join(file_manager.local_path_training_data, "false")

corrupt_images = []
total_images_checked = 0

for root, dirs, files in os.walk(root_folder):
    for filename in files:
        # Check common image extensions; skip other files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            filepath = os.path.join(root, filename)
            try:
                with Image.open(filepath) as img:
                    # If Pillow can read and verify the file, it's likely okay
                    img.verify() 
            except Exception as e:
                print(f"Corrupted file found: {filepath} — {e}")
                corrupt_images.append(filepath)
            else:
                total_images_checked += 1

print(f"Total images checked: {total_images_checked}")
print(f"Total corrupted images: {len(corrupt_images)}")
