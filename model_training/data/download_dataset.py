"""Script to download the dataset."""
import os
import yaml
from pathlib import Path
import requests
from tqdm import tqdm

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["data"]["download"]

def download_file(url: str, destination: str):
    """Download a file from url to destination with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def main():
    """Main function to download the datasets."""
    params = load_params()
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download training data
    training_data = params["training_data"]
    training_dest = raw_data_dir / training_data["dataset_name"]
    print(f"Downloading training dataset to {training_dest}")
    download_file(training_data["url"], str(training_dest))
    
    # Download test data
    test_data = params["test_data"]
    test_dest = raw_data_dir / test_data["dataset_name"]
    print(f"Downloading test dataset to {test_dest}")
    download_file(test_data["url"], str(test_dest))

if __name__ == "__main__":
    main() 