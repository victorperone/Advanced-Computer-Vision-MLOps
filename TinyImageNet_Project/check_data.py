import os

# Define the paths we expect to see
base_dir = "data/tiny-imagenet-200"
val_dir = os.path.join(base_dir, "val")

print("🔍 --- TINY IMAGENET PATH DIAGNOSTIC ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Looking for dataset at: {os.path.abspath(base_dir)}")
print("-" * 40)

# 1. Check if the base validation folder exists
if not os.path.exists(val_dir):
    print("❌ FATAL: The 'val' directory does not exist at all!")
else:
    print("✅ 'val' directory found.")
    
    # Let's see what is directly inside 'val'
    val_contents = os.listdir(val_dir)
    print(f"📂 Contents of 'val/': {val_contents}")
    
    # 2. Check for the 'images' subfolder
    if 'images' in val_contents:
        images_path = os.path.join(val_dir, 'images')
        subfolders = os.listdir(images_path)
        print(f"✅ 'val/images/' folder found.")
        print(f"📁 Number of items inside 'val/images/': {len(subfolders)}")
        
        # Check if they are folders (classes) or just flat .jpg files
        if len(subfolders) > 0:
            sample_item = os.path.join(images_path, subfolders[0])
            if os.path.isdir(sample_item):
                print(f"✨ Structure looks CORRECT (Categorical Folders found: e.g., {subfolders[0]})")
            else:
                print(f"⚠️ Structure looks FLAT (Files found instead of folders: e.g., {subfolders[0]})")
                print("   👉 You need to run 'python -m src.data_loader'!")
    else:
        print("❌ ERROR: 'images' folder is MISSING from inside 'val/'!")
        print("   👉 The folder might have been deleted, or the structure was altered.")

print("-" * 40)