import os
import random
import shutil

# --- CONFIGURATION ---
source_dir = "dataset_parking_final"
output_dir = "dataset_final_details_yolo_v2"
split_ratio = 0.8  # 80% train, 20% validation

# Create folder structure
for folder in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, folder, 'labels'), exist_ok=True)

# List images (each image has a .txt label with the same name)
images = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.jpg')]
random.shuffle(images)

# Compute split index
split_idx = int(len(images) * split_ratio)
train_files = images[:split_idx]
val_files = images[split_idx:]

def move_files(file_list, target_folder):
    for filename in file_list:
        name = os.path.splitext(filename)[0]
        
        # Copy image
        shutil.copy(os.path.join(source_dir, 'images', filename), 
                    os.path.join(output_dir, target_folder, 'images', filename))
        
        # Copy label (.txt)
        shutil.copy(os.path.join(source_dir, 'labels', name + '.txt'), 
                    os.path.join(output_dir, target_folder, 'labels', name + '.txt'))

# Run the copy
move_files(train_files, 'train')
move_files(val_files, 'val')

print(f"Done! Dataset split:")
print(f" - Train: {len(train_files)} images")
print(f" - Validation: {len(val_files)} images")