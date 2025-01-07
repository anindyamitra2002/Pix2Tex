from datasets import load_dataset, Dataset, DatasetDict
import os
from tqdm.auto import tqdm
import gc
import torch
from PIL import Image
import shutil

def process_dataset(dataset_repo: str, image_folder: str, batch_size: int = 1000):
    """
    Process a HuggingFace dataset containing images, save images to disk, and update dataset.
    
    Args:
        dataset_repo (str): HuggingFace dataset repository name
        image_folder (str): Base folder to save images
        batch_size (int): Number of examples to process at once
    """
    # Create base image folder if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(dataset_repo)
    
    # Process each split
    processed_dataset = {}
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split...")
        
        # Create split-specific folder
        split_folder = os.path.join(image_folder, split_name)
        os.makedirs(split_folder, exist_ok=True)
        
        # Initialize lists to store new data
        new_data = {
            'image': [],  # Will store image paths
            'latex': []   # Will store original latex
        }
        
        # Process the split in batches
        for i in tqdm(range(0, len(split_data), batch_size)):
            # Get batch using select
            end_idx = min(i + batch_size, len(split_data))
            batch = split_data.select(range(i, end_idx))
            
            # Get all images and latex from batch
            images = batch['image']
            latex = batch['latex']
            
            # Process each example in the batch
            for j, (img, tex) in enumerate(zip(images, latex)):
                # Generate unique filename
                global_idx = i + j
                filename = f"img_{global_idx:08d}.png"
                filepath = os.path.join(split_folder, filename)
                
                # Save image
                if isinstance(img, Image.Image):
                    img.save(filepath)
                else:
                    Image.fromarray(img).save(filepath)
                
                # Store new data
                new_data['image'].append(filepath)
                new_data['latex'].append(tex)
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Create new dataset with file paths instead of images
        processed_dataset[split_name] = Dataset.from_dict(new_data)
        
        # Clear split-level memory
        gc.collect()
    
    # Create new dataset dictionary
    final_dataset = DatasetDict(processed_dataset)
    
    # Push to HuggingFace
    try:
        final_dataset.push_to_hub(dataset_repo + "_processed")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        # Save locally as fallback
        final_dataset.save_to_disk("processed_dataset")
        print("Dataset saved locally as 'processed_dataset'")
    
    return final_dataset

def cleanup_folder(folder: str):
    """Remove a folder and its contents if it exists."""
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Example usage
if __name__ == "__main__":
    # Configuration
    DATASET_REPO = "anindya-hf-2002/pix2tex"
    IMAGE_FOLDER = "/teamspace/studios/this_studio/Pix2Tex/data"
    BATCH_SIZE = 5000  # Adjust based on available RAM

    dataset = process_dataset(DATASET_REPO, IMAGE_FOLDER, BATCH_SIZE)
    print("Processing completed successfully!")
