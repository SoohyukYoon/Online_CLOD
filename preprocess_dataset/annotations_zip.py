import os
import zipfile
from pathlib import Path
from datetime import datetime

def find_json_files(folder_path):
    """
    Find all JSON files in a folder and its subfolders.
    
    Args:
        folder_path (str): Path to the folder to search
        
    Returns:
        list: List of Path objects for all JSON files found
    """
    json_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return json_files
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return json_files
    
    # Use glob to find all .json files recursively
    json_files = list(folder.glob('**/*.json'))
    
    return json_files

def zip_individual_json_files(folder_path, output_folder=None):
    """
    Create individual ZIP files for each JSON file found in the folder and subfolders.
    
    Args:
        folder_path (str): Path to the folder to search
        output_folder (str): Folder to save ZIP files (optional, defaults to same as source)
        
    Returns:
        list: List of created ZIP file paths
    """
    
    # Find all JSON files
    json_files = find_json_files(folder_path)
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}' and its subfolders")
        return []
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        print(f"  - {file}")
    print("-" * 50)
    
    # Set output folder
    if output_folder is None:
        output_folder = folder_path
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_zips = []
    
    try:
        for json_file in json_files:
            # Create ZIP filename based on JSON filename
            json_name = json_file.stem  # filename without extension
            zip_filename = f"{json_name}.zip"
            
            # If preserving folder structure, create subdirectories in output
            relative_path = json_file.relative_to(Path(folder_path))
            zip_output_dir = output_path / relative_path.parent
            zip_output_dir.mkdir(parents=True, exist_ok=True)
            
            zip_file_path = zip_output_dir / zip_filename
            
            # Overwrite existing ZIP file if it exists
            if zip_file_path.exists():
                print(f"Overwriting existing: {zip_file_path}")

            # Create ZIP file containing the single JSON file
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(json_file, json_file.name)
            
            created_zips.append(zip_file_path)
            
            # Show file sizes
            original_size = json_file.stat().st_size
            zip_size = zip_file_path.stat().st_size
            compression_ratio = (1 - zip_size / original_size) * 100 if original_size > 0 else 0
            
            print(f"Created: {zip_file_path}")
            print(f"  Original: {original_size} bytes -> Compressed: {zip_size} bytes ({compression_ratio:.1f}% compression)")
        
        print(f"\nSuccessfully created {len(created_zips)} ZIP files")
        return created_zips
        
    except Exception as e:
        print(f"Error creating ZIP files: {e}")
        return created_zips

def zip_individual_json_files_flat(folder_path, output_folder=None):
    """
    Create individual ZIP files for each JSON file, saving all ZIPs in a single output folder.
    
    Args:
        folder_path (str): Path to the folder to search
        output_folder (str): Folder to save all ZIP files (optional)
        
    Returns:
        list: List of created ZIP file paths
    """
    
    # Find all JSON files
    json_files = find_json_files(folder_path)
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}' and its subfolders")
        return []
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        print(f"  - {file}")
    print("-" * 50)
    
    # Set output folder
    if output_folder is None:
        base_folder = Path(folder_path).name
        output_folder = f"{base_folder}_zipped_jsons"
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_zips = []
    
    try:
        for json_file in json_files:
            # Create unique ZIP filename
            json_name = json_file.stem
            
            # Add parent folder info to avoid naming conflicts
            relative_path = json_file.relative_to(Path(folder_path))
            if len(relative_path.parts) > 1:
                parent_dirs = "_".join(relative_path.parts[:-1])
                zip_filename = f"{parent_dirs}_{json_name}.zip"
            else:
                zip_filename = f"{json_name}.zip"
            
            zip_file_path = output_path / zip_filename
            
            # Overwrite existing ZIP file if it exists
            if zip_file_path.exists():
                print(f"Overwriting existing: {zip_file_path}")
            
            # Create ZIP file containing the single JSON file
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(json_file, json_file.name)
            
            created_zips.append(zip_file_path)
            
            # Show file sizes
            original_size = json_file.stat().st_size
            zip_size = zip_file_path.stat().st_size
            compression_ratio = (1 - zip_size / original_size) * 100 if original_size > 0 else 0
            
            print(f"Created: {zip_file_path}")
            print(f"  Original: {original_size} bytes -> Compressed: {zip_size} bytes ({compression_ratio:.1f}% compression)")
        
        print(f"\nSuccessfully created {len(created_zips)} ZIP files in '{output_path}'")
        return created_zips
        
    except Exception as e:
        print(f"Error creating ZIP files: {e}")
        return created_zips

def list_created_zips(zip_files):
    """
    List information about the created ZIP files.
    
    Args:
        zip_files (list): List of ZIP file paths
    """
    if not zip_files:
        print("No ZIP files to list")
        return
    
    print(f"\nSummary of {len(zip_files)} created ZIP files:")
    print("-" * 60)
    
    total_original_size = 0
    total_zip_size = 0
    
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zipf:
                info_list = zipf.infolist()
                if info_list:
                    original_size = info_list[0].file_size
                    zip_size = zip_file.stat().st_size
                    total_original_size += original_size
                    total_zip_size += zip_size
                    
                    print(f"{zip_file.name}: {original_size} -> {zip_size} bytes")
        except Exception as e:
            print(f"Error reading {zip_file}: {e}")
    
    if total_original_size > 0:
        overall_compression = (1 - total_zip_size / total_original_size) * 100
        print(f"\nOverall: {total_original_size} -> {total_zip_size} bytes ({overall_compression:.1f}% compression)")

def main():
    """Main function using local variables"""
    
    # Configuration - modify these variables as needed
    folder_to_search = "./data"          # Path to folder to search for JSON files
    output_folder = None                 # Output folder for ZIP files (None for same as source)
    preserve_folder_structure = True     # Whether to preserve folder structure for ZIP locations
    show_zip_summary = True              # Whether to show summary of created ZIPs
    
    print(f"Searching for JSON files in: {folder_to_search}")
    print(f"Preserve folder structure: {preserve_folder_structure}")
    print(f"Output folder: {output_folder or 'Same as source folders'}")
    print("=" * 60)
    
    # Create individual ZIP files for each JSON file
    if preserve_folder_structure:
        zip_files = zip_individual_json_files(
            folder_path=folder_to_search,
            output_folder=output_folder
        )
    else:
        zip_files = zip_individual_json_files_flat(
            folder_path=folder_to_search,
            output_folder=output_folder
        )
    
    # Show summary if requested
    if zip_files and show_zip_summary:
        list_created_zips(zip_files)

def example_usage():
    """Example of how to use the functions"""
    
    # Example 1: Preserve folder structure
    print("Example 1: Preserving folder structure")
    zip_files1 = zip_individual_json_files("./data")
    
    # Example 2: All ZIPs in one folder
    print("\nExample 2: All ZIPs in one output folder")
    zip_files2 = zip_individual_json_files_flat(
        folder_path="./data",
        output_folder="./all_json_zips"
    )
    
    # Example 3: Custom output folder with structure preserved
    print("\nExample 3: Custom output folder with structure")
    zip_files3 = zip_individual_json_files(
        folder_path="./data",
        output_folder="./backup_zips"
    )
    
    # Show summaries
    if zip_files2:
        list_created_zips(zip_files2)

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment the line below to run examples instead
    # example_usage()