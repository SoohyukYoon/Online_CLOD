# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Download (if needed), extract, and flatten the SHIFT **val** image archive.

# After the script finishes, you will have:
# ./data/shift/images/val/  (containing validation images)
# """
# import os
# import shutil
# import zipfile
# import requests
# from tqdm import tqdm

# # --------------------------------------------------------------------------- #
# # UTILITIES                                                                   #
# # --------------------------------------------------------------------------- #
# def file_is_complete(local_path: str, remote_url: str) -> bool:
#     if not os.path.exists(local_path):
#         return False
#     try:
#         r = requests.head(remote_url, timeout=10)
#         r.raise_for_status()
#         return os.path.getsize(local_path) == int(r.headers.get("content-length", 0))
#     except Exception:
#         # If we cannot verify, assume the file is OK.
#         return True


# def download(url: str, dst: str) -> None:
#     """Stream-download *url* into *dst* with a tqdm progress bar."""
#     os.makedirs(os.path.dirname(dst), exist_ok=True)

#     head = requests.head(url, timeout=10)
#     head.raise_for_status()
#     total = int(head.headers.get("content-length", 0))

#     with requests.get(url, stream=True, timeout=30) as r, \
#          open(dst, "wb") as f,                                    \
#          tqdm(total=total, unit="B", unit_scale=True,
#               desc=os.path.basename(dst)) as bar:
#         r.raise_for_status()
#         for chunk in r.iter_content(chunk_size=8192):
#             if chunk:                                   # skip keep-alive
#                 f.write(chunk)
#                 bar.update(len(chunk))


# def extract_and_flatten(zip_file: str, target_dir: str) -> None:
#     """
#     Extract ZIP and flatten all images directly into target_dir.
#     Handle filename conflicts by prefixing with subdirectory name.
#     """
#     print(f"\nExtracting and flattening {os.path.basename(zip_file)} ...")
    
#     # Create target directory
#     os.makedirs(target_dir, exist_ok=True)
    
#     with zipfile.ZipFile(zip_file) as z:
#         image_files = [f for f in z.namelist() 
#                       if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.endswith('/')]
        
#         print(f"Found {len(image_files)} image files to extract...")
        
#         with tqdm(total=len(image_files), unit="files") as bar:
#             for member in image_files:
#                 # Create unique filename by combining subdir and filename
#                 # e.g., "c6ae-003a/00000000_img_front.jpg" -> "c6ae-003a_00000000_img_front.jpg"
#                 parts = member.split('/')
#                 if len(parts) > 1:
#                     subdir = parts[0]
#                     filename = parts[-1]
#                     unique_filename = f"{subdir}_{filename}"
#                 else:
#                     unique_filename = parts[0]
                
#                 target_path = os.path.join(target_dir, unique_filename)
                
#                 # Extract the file data and write directly to target
#                 try:
#                     with z.open(member) as source, open(target_path, 'wb') as target:
#                         shutil.copyfileobj(source, target)
#                 except Exception as e:
#                     print(f"Error extracting {member}: {e}")
#                     continue
                    
#                 bar.update(1)


# def move_into_place(src_dir: str, dst_dir: str) -> None:
#     """Move *src_dir* wholesale to *dst_dir*, replacing existing dir if needed."""
#     if os.path.exists(dst_dir):
#         print(f"Overwriting existing {dst_dir}")
#         shutil.rmtree(dst_dir)
#     os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
#     shutil.move(src_dir, dst_dir)


# # --------------------------------------------------------------------------- #
# # MAIN                                                                        #
# # --------------------------------------------------------------------------- #
# def main() -> None:

#     # --------------------------------------------------------------------------- #
#     # CONFIGURATION                                                               #
#     # --------------------------------------------------------------------------- #
#     url         = "https://dl.cv.ethz.ch/shift/discrete/images/val/front/img.zip"
#     root_dir    = "./data/shift"                      # everything lives under here
#     zip_path    = os.path.join(root_dir, "img.zip")
#     final_dir   = os.path.join(root_dir, "images", "val")           # <- target

#     # 1. Download (if necessary)
#     if not file_is_complete(zip_path, url):
#         print(f"Downloading {url} …")
#         download(url, zip_path)
#     else:
#         print(f"Found complete archive: {zip_path}")

#     # 2. Extract and flatten images directly to final directory
#     extract_and_flatten(zip_path, final_dir)

#     print(f"\nFinished!  Images are now in: {final_dir}")


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nInterrupted by user – exiting.")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download (if needed), extract, and flatten the SHIFT **val** and **train** image archives.

After the script finishes, you will have:
./data/shift/images/val/  (containing validation images)
./data/shift/images/train/  (containing training images)
"""
import os
import shutil
import zipfile
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# UTILITIES                                                                   #
# --------------------------------------------------------------------------- #
def file_is_complete(local_path: str, remote_url: str) -> bool:
    if not os.path.exists(local_path):
        return False
    try:
        r = requests.head(remote_url, timeout=10)
        r.raise_for_status()
        return os.path.getsize(local_path) == int(r.headers.get("content-length", 0))
    except Exception:
        # If we cannot verify, assume the file is OK.
        return True


def download(url: str, dst: str) -> None:
    """Stream-download *url* into *dst* with a tqdm progress bar."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    head = requests.head(url, timeout=10)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))

    with requests.get(url, stream=True, timeout=30) as r, \
         open(dst, "wb") as f,                                    \
         tqdm(total=total, unit="B", unit_scale=True,
              desc=os.path.basename(dst)) as bar:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:                                   # skip keep-alive
                f.write(chunk)
                bar.update(len(chunk))


def extract_and_flatten(zip_file: str, target_dir: str) -> None:
    """
    Extract ZIP and flatten all images directly into target_dir.
    Handle filename conflicts by prefixing with subdirectory name.
    """
    print(f"\nExtracting and flattening {os.path.basename(zip_file)} ...")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_file) as z:
        image_files = [f for f in z.namelist() 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.endswith('/')]
        
        print(f"Found {len(image_files)} image files to extract...")
        
        with tqdm(total=len(image_files), unit="files") as bar:
            for member in image_files:
                # Create unique filename by combining subdir and filename
                # e.g., "c6ae-003a/00000000_img_front.jpg" -> "c6ae-003a_00000000_img_front.jpg"
                parts = member.split('/')
                if len(parts) > 1:
                    subdir = parts[0]
                    filename = parts[-1]
                    unique_filename = f"{subdir}_{filename}"
                else:
                    unique_filename = parts[0]
                
                target_path = os.path.join(target_dir, unique_filename)
                
                # Extract the file data and write directly to target
                try:
                    with z.open(member) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                except Exception as e:
                    print(f"Error extracting {member}: {e}")
                    continue
                    
                bar.update(1)


def move_into_place(src_dir: str, dst_dir: str) -> None:
    """Move *src_dir* wholesale to *dst_dir*, replacing existing dir if needed."""
    if os.path.exists(dst_dir):
        print(f"Overwriting existing {dst_dir}")
        shutil.rmtree(dst_dir)
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.move(src_dir, dst_dir)


# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:

    # --------------------------------------------------------------------------- #
    # CONFIGURATION                                                               #
    # --------------------------------------------------------------------------- #
    base_url    = "https://dl.cv.ethz.ch/shift/discrete/images"
    root_dir    = "./data/shift"                      # everything lives under here
    
    # Define datasets to download (val first, then train)
    datasets = [
        {"subset": "val", "url": f"{base_url}/val/front/img.zip"},
        {"subset": "train", "url": f"{base_url}/train/front/img.zip"}
    ]
    
    for dataset in datasets:
        subset = dataset["subset"]
        url = dataset["url"]
        
        print(f"\n{'='*60}")
        print(f"Processing {subset.upper()} dataset")
        print(f"{'='*60}")
        
        zip_path = os.path.join(root_dir, f"{subset}_img.zip")
        final_dir = os.path.join(root_dir, "images", subset)
        
        # 1. Download (if necessary)
        if not file_is_complete(zip_path, url):
            print(f"Downloading {url} …")
            download(url, zip_path)
        else:
            print(f"Found complete archive: {zip_path}")

        # 2. Extract and flatten images directly to final directory
        extract_and_flatten(zip_path, final_dir)

        print(f"\nFinished {subset}! Images are now in: {final_dir}")
    
    print(f"\n{'='*60}")
    print("ALL DOWNLOADS COMPLETE!")
    print(f"Val images: ./data/shift/images/val/")
    print(f"Train images: ./data/shift/images/train/")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting.")