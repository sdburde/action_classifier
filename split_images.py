import os
import shutil
from sklearn.model_selection import train_test_split
import uuid  # For generating unique file names

# Paths to the main dataset folder and where the split data will be stored
main_dataset_dir = "action_crop_dataset"
output_dir = "split_dataset"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "val")

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all class folders in the main dataset
class_folders = os.listdir(main_dataset_dir)

# Function to split, copy, and rename files
def split_class_data(class_name):
    class_path = os.path.join(main_dataset_dir, class_name)
    images = os.listdir(class_path)
    
    # Split into train (78%) and test (22%)
    train_images, test_images = train_test_split(images, test_size=0.22, random_state=42, shuffle=True)

    # Create class directories in train and test folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Copy and rename training images
    for img in train_images:
        src_path = os.path.join(class_path, img)
        unique_name = f"{uuid.uuid4()}.jpg"  # Generate a unique name with a .jpg extension
        dst_path = os.path.join(train_dir, class_name, unique_name)
        shutil.copy(src_path, dst_path)

    # Copy and rename test images
    for img in test_images:
        src_path = os.path.join(class_path, img)
        unique_name = f"{uuid.uuid4()}.jpg"  # Generate a unique name with a .jpg extension
        dst_path = os.path.join(test_dir, class_name, unique_name)
        shutil.copy(src_path, dst_path)

    # Print the number of images in train and test for each class
    print(f"Class: {class_name}")
    print(f"Train: {len(train_images)} images")
    print(f"Test: {len(test_images)} images\n")

# Iterate over each class folder and split the images
for class_folder in class_folders:
    split_class_data(class_folder)

print("Dataset split into train and test folders successfully with images shuffled and renamed.")
