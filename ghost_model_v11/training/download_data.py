
"""
Ghost v11 - Offline Data Downloader
===================================
Downloads the 'roneneldan/TinyStories' dataset for offline training.
"""

import os
from datasets import load_dataset

def download_data():
    dataset_name = "roneneldan/TinyStories"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/tinystories")
    
    print(f"📥 Downloading {dataset_name}...")
    print(f"📂 Output Dir: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and save to disk
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(output_dir)
    
    print("✅ Download Complete!")
    print(f"   Usage: dataset = load_from_disk('{output_dir}')")

def download_openwebtext():
    url = "https://hub.oxen.ai/api/repos/Elriggs/openwebtext-100k/file/main/openwebtext-100k_train.parquet"
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/openwebtext.parquet")
    
    print(f"📥 Downloading OpenWebText Parquet...")
    print(f"🔗 URL: {url}")
    print(f"📂 Output: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.system(f"curl -L -o {output_path} {url}")
    
    print("✅ Download Complete!")

if __name__ == "__main__":
    # download_data() # TinyStories
    download_openwebtext() # 100k samples
