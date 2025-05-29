# import requests
# import os
# from tqdm import tqdm
# import time
# import zipfile
# import shutil

# def file_exists_and_valid(file_path, url=None):
#     """
#     Check if a file exists and optionally verify its size against the remote file.
    
#     Args:
#         file_path (str): Path to the local file
#         url (str, optional): URL of the remote file to check size against
        
#     Returns:
#         bool: True if file exists (and size matches if URL provided), False otherwise
#     """
#     if not os.path.exists(file_path):
#         return False
    
#     # If no URL provided, just check existence
#     if url is None:
#         return True
    
#     try:
#         # Get local file size
#         local_size = os.path.getsize(file_path)
        
#         # Get remote file size
#         response = requests.head(url, timeout=10)
#         remote_size = int(response.headers.get('content-length', 0))
        
#         # If remote size is 0, we can't verify
#         if remote_size == 0:
#             print(f"Warning: Couldn't verify remote file size, assuming local file is valid.")
#             return True
        
#         # Compare sizes
#         if local_size == remote_size:
#             return True
#         else:
#             print(f"Warning: Local file size ({local_size} bytes) doesn't match remote file size ({remote_size} bytes).")
#             return False
    
#     except requests.exceptions.RequestException as e:
#         print(f"Warning: Couldn't verify remote file size: {e}")
#         # If we can't check the remote size, assume local file is valid
#         return True

# def download_file(url, output_path):
#     """
#     Download a file from a URL to the specified output path with progress tracking.
    
#     Args:
#         url (str): URL to download from
#         output_path (str): Path where the file will be saved
#     """
#     try:
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         # Send a HEAD request first to get the file size
#         response = requests.head(url, timeout=10)
#         file_size = int(response.headers.get('content-length', 0))
        
#         # Start the download with a stream
#         response = requests.get(url, stream=True, timeout=30)
#         response.raise_for_status()  # Raise an exception for HTTP errors
        
#         # Create a progress bar
#         progress_bar = tqdm(
#             total=file_size, 
#             unit='B', 
#             unit_scale=True, 
#             desc=os.path.basename(output_path)
#         )
        
#         # Open the output file and write chunks
#         with open(output_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:  # Filter out keep-alive chunks
#                     file.write(chunk)
#                     progress_bar.update(len(chunk))
        
#         progress_bar.close()
        
#         # Verify the download is complete
#         actual_size = os.path.getsize(output_path)
#         if file_size > 0 and actual_size != file_size:
#             print(f"Warning: Downloaded file size ({actual_size} bytes) does not match expected size ({file_size} bytes)")
#             return False
#         else:
#             print(f"\nDownload complete: {output_path}")
#             return True
            
#     except requests.exceptions.RequestException as e:
#         print(f"Download error: {e}")
#         return False
#     except KeyboardInterrupt:
#         print("\nDownload canceled by user")
#         # Remove the partially downloaded file
#         if os.path.exists(output_path):
#             os.remove(output_path)
#         print(f"Partially downloaded file removed: {output_path}")
#         return False

# def extract_zip(zip_path, extract_dir):
#     """
#     Extract a zip file with progress tracking.
    
#     Args:
#         zip_path (str): Path to the zip file
#         extract_dir (str): Directory to extract files to
#     """
#     try:
#         # Create extraction directory if it doesn't exist
#         os.makedirs(extract_dir, exist_ok=True)
        
#         # Get the total size for progress reporting
#         zip_size = os.path.getsize(zip_path)
        
#         print(f"\nExtracting {os.path.basename(zip_path)} to {extract_dir}")
        
#         # Open the zip file
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             # Get list of files to extract
#             file_list = zip_ref.namelist()
#             total_files = len(file_list)
            
#             # Set up progress bar
#             progress_bar = tqdm(total=total_files, unit='files')
            
#             # Extract each file
#             for i, file in enumerate(file_list):
#                 zip_ref.extract(file, extract_dir)
#                 progress_bar.update(1)
            
#             progress_bar.close()
        
#         print(f"Extraction complete: {len(file_list)} files extracted to {extract_dir}")
#         return True
    
#     except zipfile.BadZipFile:
#         print(f"Error: {zip_path} is not a valid zip file")
#         return False
#     except Exception as e:
#         print(f"Extraction error: {e}")
#         return False
#     except KeyboardInterrupt:
#         print("\nExtraction canceled by user")
#         # Consider cleaning up partial extraction
#         print(f"Note: Some files may have been partially extracted to {extract_dir}")
#         return False

# def reorganize_folders(extract_dir, parent_dir):
#     """
#     Reorganize folder structure by moving subdirectories up one level.
    
#     Args:
#         extract_dir (str): Path to the extracted directory that contains subdirectories
#         parent_dir (str): Path to the parent directory where subdirectories should be moved
#     """
#     try:
#         print(f"\nReorganizing folder structure...")
        
#         # Check if extract directory exists
#         if not os.path.exists(extract_dir):
#             print(f"Error: Extracted directory {extract_dir} does not exist")
#             return False
        
#         # Get list of subdirectories in extract_dir
#         subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
        
#         if not subdirs:
#             print(f"No subdirectories found in {extract_dir}")
#             return False
        
#         # Create parent directory if it doesn't exist
#         os.makedirs(parent_dir, exist_ok=True)
        
#         # Move each subdirectory to parent directory
#         for subdir in subdirs:
#             src_path = os.path.join(extract_dir, subdir)
#             dst_path = os.path.join(parent_dir, subdir)
            
#             # If destination already exists, ask for confirmation
#             if os.path.exists(dst_path):
#                 choice = input(f"{dst_path} already exists. Replace? (y/n): ").lower()
#                 if choice != 'y':
#                     print(f"Skipping {subdir}")
#                     continue
#                 # If replacing, remove existing directory
#                 shutil.rmtree(dst_path)
            
#             # Move directory
#             print(f"Moving {subdir} to {parent_dir}")
#             shutil.move(src_path, dst_path)
        
#         # Check if extract_dir is now empty
#         remaining_files = os.listdir(extract_dir)
#         if not remaining_files:
#             # Remove empty directory
#             os.rmdir(extract_dir)
#             print(f"Removed empty directory: {extract_dir}")
#         else:
#             print(f"Note: {extract_dir} still contains {len(remaining_files)} files/folders")
        
#         print(f"Folder reorganization complete!")
#         return True
        
#     except Exception as e:
#         print(f"Error reorganizing folders: {e}")
#         return False

# if __name__ == "__main__":
    
#     url = "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"
#     # url = "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
#     # url = "http://128.32.162.150/bdd100k/bdd100k_images_10k.zip"
#     output_dir = "./data/bdd100k"
#     filename = os.path.basename(url)
#     output_path = os.path.join(output_dir, filename)
    
#     # Directory where the files will be extracted initially
#     extract_dir = os.path.join(output_dir, 'images/bdd100k/images/100k/val')
    
#     # Parent directory where we want train, val, test folders to end up
#     parent_dir = os.path.join(output_dir, "images")
    
#     # Check if file already exists
#     if file_exists_and_valid(output_path, url):
#         print(f"File already exists: {output_path}")
#         download_success = True
        
#         # Ask if user wants to force redownload
#         redownload = input("Do you want to redownload the file? (y/n): ").lower()
#         if redownload == 'y':
#             print(f"Starting redownload of {filename}")
#             download_success = download_file(url, output_path)
#     else:
#         print(f"Starting download of {filename}")
#         print(f"This is a large dataset file and may take some time to download")
#         download_success = download_file(url, output_path)
    
#     # Extract only if download was successful
#     if download_success:
#         # Check if extraction directory already exists and has content
#         if os.path.exists(extract_dir) and os.listdir(extract_dir):
#             print(f"Extraction directory already exists and contains files: {extract_dir}")
#             extract_again = input("Do you want to extract again? (y/n): ").lower()
#             if extract_again == 'y':
#                 extract_success = extract_zip(output_path, extract_dir)
#             else:
#                 extract_success = True
#         else:
#             extract_success = extract_zip(output_path, extract_dir)
        

#         if extract_success:

#             # locate the train/val/test folders, no matter how deep
#             for candidate in [
#                 os.path.join(extract_dir, "images", "100k"),
#                 os.path.join(extract_dir, "100k"),
#                 extract_dir
#             ]:
#                 if os.path.isdir(candidate):
#                     subdir_to_check = candidate
#                     break

#             reorganize_success = reorganize_folders(subdir_to_check, parent_dir)

#             if reorganize_success:
#                 print("\nImages now live in:", parent_dir)
#                 shutil.rmtree(extract_dir, ignore_errors=True)   # delete scratch



#         if extract_success:
#             # Reorganize folders: move train, val, test to parent folder
#             subdir_to_check = os.path.join(extract_dir, "100k")  # Assuming the 10k folder is inside extract_dir
            
#             # If 10k folder exists, use it as the source for reorganization
#             if os.path.exists(subdir_to_check) and os.path.isdir(subdir_to_check):
#                 reorganize_success = reorganize_folders(subdir_to_check, parent_dir)
#             else:
#                 # Otherwise, use the extract_dir itself
#                 reorganize_success = reorganize_folders(extract_dir, parent_dir)
            
#             if reorganize_success:
#                 print(f"\nDownload, extraction and reorganization complete!")
#                 print(f"Files are available at: {parent_dir}")
            
#             # Optionally, ask if the user wants to delete the zip file
#             if os.path.exists(output_path):
#                 delete_zip = input("\nDo you want to delete the zip file to save space? (y/n): ").lower()
#                 if delete_zip == 'y':
#                     os.remove(output_path)
#                     print(f"Zip file deleted: {output_path}")


#     else:
#         print("Download failed. Extraction skipped.")





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download (if needed), extract, and flatten the BDD100K **val** image archive.

After the script finishes, you will have:
./data/bdd100k/images/val/  (containing 20 k validation images)

If you later grab the *train* or *test* archives, just change the `subset`
variable near the top.
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


def extract(zip_file: str, to_dir: str) -> str:
    """
    Extract ZIP to *to_dir* and return the **deepest path that contains the
    actual images**.  For the BDD100K archives this is always
    .../images/bdd100k/images/100k/<subset>
    """
    print(f"\nExtracting {os.path.basename(zip_file)} ...")
    if os.path.exists(to_dir):
        shutil.rmtree(to_dir)
    os.makedirs(to_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file) as z, \
         tqdm(total=len(z.namelist()), unit="files") as bar:
        for member in z.namelist():
            z.extract(member, to_dir)
            bar.update(1)

    # autodetect the “deepest” folder that actually holds image files
    for root, dirs, files in os.walk(to_dir):
        if files and all(f.lower().endswith(".jpg") for f in files):
            return root                 # ← e.g. .../100k/val
    raise RuntimeError("Could not locate extracted images!")


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
    subsets      = ["train", "val"]

    for subset in subsets:
        url         = f"https://dl.cv.ethz.ch/bdd100k/data/100k_images_{subset}.zip"
        root_dir    = "./data/bdd100k"            # everything lives under here
        zip_path    = os.path.join(root_dir, os.path.basename(url))
        final_dir   = os.path.join(root_dir, "images", subset)        # <- target
        scratch_dir = os.path.join(root_dir, "_extract_tmp")          # <- scratch

        # 1. Download (if necessary)
        if not file_is_complete(zip_path, url):
            print(f"Downloading {url} …")
            download(url, zip_path)
        else:
            print(f"Found complete archive: {zip_path}")

        # 2. Extract to scratch dir
        extracted_leaf = extract(zip_path, scratch_dir)

        # 3. Move images/<subset> to the final flat location
        move_into_place(extracted_leaf, final_dir)

        # 4. Clean up
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
        print(f"\nFinished!  Images are now in: {final_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting.")