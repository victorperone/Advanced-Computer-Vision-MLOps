import os
import shutil

# --- CONFIGURATION ---
# We point to where the 'val' data currently sits
base_val_dir = 'data/tiny-imagenet-200/val'
images_dir = os.path.join(base_val_dir, 'images')
annotations_file = os.path.join(base_val_dir, 'val_annotations.txt')

def organize_validation_data():
    """
    This function reads the text file that tells us which image belongs to which class,
    creates a folder for that class, and moves the image inside it.
    """
    
    # 1. Safety Check: Does the annotations file exist?
    if not os.path.exists(annotations_file):
        print(f"ERROR: Could not find {annotations_file}. Are you in the project root?")
        return

    # 2. Open the 'map' (the text file)
    # This file has lines like: val_0.jpg  n03444034  0  0  64  64
    print("Step 5: Reading val_annotations.txt...")
    with open(annotations_file, 'r') as f:
        lines = f.readlines()

    # 3. Loop through every image mentioned in the file
    print(f"Processing {len(lines)} validation images...")
    for line in lines:
        # Split the line by tabs
        parts = line.split('\t')
        image_name = parts[0]   # Example: 'val_0.jpg'
        class_id = parts[1]     # Example: 'n03444034'

        # 4. Create a folder for this class if it doesn't exist yet
        # New path: data/tiny-imagenet-200/val/images/n03444034/
        class_folder_path = os.path.join(images_dir, class_id)
        
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

        # 5. Define where the image is NOW and where it SHOULD GO
        current_location = os.path.join(images_dir, image_name)
        new_location = os.path.join(class_folder_path, image_name)

        # 6. Move the image
        if os.path.exists(current_location):
            shutil.move(current_location, new_location)

    print("✅ SUCCESS: Validation folder is now organized by class folders!")

if __name__ == "__main__":
    organize_validation_data()
