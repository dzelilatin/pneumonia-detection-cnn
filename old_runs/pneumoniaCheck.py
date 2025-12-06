import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- VS CODE AUTOMATIC SETUP ---
# This finds your home folder (e.g. /Users/yourname) automatically
home_folder = os.path.expanduser("~")
base_dir = os.path.join(home_folder, 'Downloads', 'chest_xray')

print(f"Looking for data at: {base_dir}")

if not os.path.exists(base_dir):
    print("❌ ERROR: Cannot find the 'chest_xray' folder.")
    print("Please go to your Downloads folder and unzip 'chest_xray.zip' so it becomes a normal folder.")
else:
    print(f"✅ FOUND IT! Reading images...")
    
    # 1. Count the images
    print("\n--- Dataset Statistics ---")
    total_normal = 0
    total_pneu = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        normal_dir = os.path.join(split_dir, 'NORMAL')
        pneu_dir = os.path.join(split_dir, 'PNEUMONIA')
        
        # Check if folders exist before counting
        if os.path.exists(normal_dir):
            n_normal = len(os.listdir(normal_dir))
            n_pneu = len(os.listdir(pneu_dir))
            total = n_normal + n_pneu
            
            total_normal += n_normal
            total_pneu += n_pneu
            
            print(f"{split.capitalize():<10} | Normal: {n_normal:<5} | Pneumonia: {n_pneu:<5} | Total: {total:<5}")
    
    print("-" * 50)
    print(f"{'Grand Total':<10} | Normal: {total_normal:<5} | Pneumonia: {total_pneu:<5} | Total: {total_normal + total_pneu:<5}")

    # 2. Show the images
    print("\nOpening image plot...")
    train_dir = os.path.join(base_dir, 'train')
    if os.path.exists(train_dir):
        # Find first image in each folder
        # We filter for .jpeg or .png to avoid hidden system files like .DS_Store
        normal_files = [f for f in os.listdir(os.path.join(train_dir, 'NORMAL')) if f.endswith('jpeg') or f.endswith('png')]
        pneu_files = [f for f in os.listdir(os.path.join(train_dir, 'PNEUMONIA')) if f.endswith('jpeg') or f.endswith('png')]

        if normal_files and pneu_files:
            normal_path = os.path.join(train_dir, 'NORMAL', normal_files[0])
            pneu_path = os.path.join(train_dir, 'PNEUMONIA', pneu_files[0])

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(mpimg.imread(normal_path), cmap='gray')
            axes[0].set_title("Normal Case")
            axes[0].axis('off')
            
            axes[1].imshow(mpimg.imread(pneu_path), cmap='gray')
            axes[1].set_title("Pneumonia Case")
            axes[1].axis('off')
            
            plt.show()
            print("Done!")