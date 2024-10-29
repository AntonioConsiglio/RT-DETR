import os
import hashlib
import requests
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm  # Progress bar library

# Dataset information
DATASET_INFO = {
    'coco': (
        [
            ('http://images.cocodataset.org/zips/train2017.zip', 'cced6f7f71b7629ddf16f17bbcfab6b2'),
            ('http://images.cocodataset.org/zips/val2017.zip', '442b8da7639aecaf257c1dceb8ba8c80'),
            ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'f4bbac642086de4f52a3fdda2de5fa2c')
        ],
        ["annotations", "train2017", "val2017"]
    )
}

# Download directory
DOWNLOAD_DIR = Path(__file__).parent.resolve()

# Helper function to compute MD5 checksum
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Helper function to download a file with checksum verification
def download_file(url, checksum, download_path):
    filename = download_path / Path(url).name

    # Skip download if file exists with correct checksum
    if filename.exists():
        print(f"{filename} already exists. Verifying checksum...")
        if calculate_md5(filename) == checksum:
            print("Checksum verified. Skipping download.")
            return filename
        else:
            print("Checksum mismatch. Re-downloading file.")

    # Start the download with a progress bar
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error if download failed

    # Get total file size from the response headers
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as f, tqdm(
        desc=filename.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    # Verify checksum
    if calculate_md5(filename) != checksum:
        raise ValueError(f"Checksum mismatch for {filename}. Download might be corrupted.")
    print(f"Downloaded and verified {filename}")
    return filename

# Main function to download and extract dataset
def download_and_extract_dataset(dataset_name):
    urls, folders = DATASET_INFO[dataset_name]
    download_path = DOWNLOAD_DIR / dataset_name
    download_path.mkdir(parents=True, exist_ok=True)

    for url, checksum in urls:
        file_path = download_file(url, checksum, download_path)
        
        # Extract if zip file
        if file_path.suffix == '.zip':
            with ZipFile(file_path, 'r') as zip_ref:
                print(f"Extracting {file_path}...")
                zip_ref.extractall(download_path)
            print(f"Extracted {file_path}")

# Run the download and extraction process
if __name__ == "__main__":
    download_and_extract_dataset('coco')
