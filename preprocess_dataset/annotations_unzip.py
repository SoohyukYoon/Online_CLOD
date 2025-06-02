import os
import zipfile
import json
from pathlib import Path
from datetime import datetime

def find_zip_files(folder_path, name_filter=None):
    """
    Find all ZIP files in a folder and its subfolders, optionally filtered by name.
    
    Args:
        folder_path (str): Path to the folder to search
        name_filter (str): Only include ZIP files whose names start with this string
        
    Returns:
        list: List of Path objects for all ZIP files found
    """
    zip_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return zip_files
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return zip_files
    
    # Use glob to find all .zip files recursively
    all_zip_files = list(folder.glob('**/*.zip'))
    
    # Filter by name if specified
    if name_filter:
        zip_files = [zf for zf in all_zip_files if zf.stem.startswith(name_filter)]
        print(f"Filtered ZIP files (starting with '{name_filter}'): {len(zip_files)} out of {len(all_zip_files)} total ZIP files")
    else:
        zip_files = all_zip_files
    
    return zip_files

def is_json_content(content):
    """
    Check if the content is valid JSON.
    
    Args:
        content (str): Content to check
        
    Returns:
        bool: True if content is valid JSON, False otherwise
    """
    try:
        json.loads(content)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def extract_json_from_zip(zip_file_path, extract_to_same_folder=True):
    """
    Extract JSON files from a single ZIP file to the same folder as the ZIP file.
    
    Args:
        zip_file_path (Path): Path to the ZIP file
        extract_to_same_folder (bool): If True, extract to same folder as ZIP file
        
    Returns:
        list: List of extracted JSON file paths
    """
    extracted_files = []
    
    # Extract to the same folder as the ZIP file
    if extract_to_same_folder:
        output_folder = zip_file_path.parent
    else:
        output_folder = zip_file_path.parent / "extracted"
        output_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zipf:
            file_list = zipf.namelist()
            
            for file_name in file_list:
                # Read file content
                file_content = zipf.read(file_name).decode('utf-8')
                
                # Check if it's JSON content
                if is_json_content(file_content):
                    # Create output filename
                    if file_name.endswith('.json'):
                        json_filename = file_name
                    else:
                        # Add .json extension if not present
                        json_filename = f"{Path(file_name).stem}.json"
                    
                    output_file_path = output_folder / json_filename
                    
                    # Overwrite existing file if it exists
                    if output_file_path.exists():
                        print(f"Overwriting existing: {output_file_path}")
                    
                    # Write JSON file
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        # Pretty-print JSON for better readability
                        json_data = json.loads(file_content)
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    extracted_files.append(output_file_path)
                    print(f"Extracted JSON: {zip_file_path.name} -> {output_file_path}")
                
                else:
                    print(f"Skipped non-JSON file: {file_name} in {zip_file_path.name}")
    
    except Exception as e:
        print(f"Error extracting from {zip_file_path}: {e}")
    
    return extracted_files

def extract_all_json_from_zips(folder_path, name_filter=None):
    """
    Extract JSON files from all ZIP files found in folder and subfolders.
    Each ZIP file is extracted to its own original folder/subfolder location.
    
    Args:
        folder_path (str): Path to the folder to search for ZIP files
        name_filter (str): Only process ZIP files whose names start with this string
        
    Returns:
        list: List of all extracted JSON file paths
    """
    
    # Find all ZIP files with optional name filter
    zip_files = find_zip_files(folder_path, name_filter)
    
    if not zip_files:
        if name_filter:
            print(f"No ZIP files starting with '{name_filter}' found in '{folder_path}' and its subfolders")
        else:
            print(f"No ZIP files found in '{folder_path}' and its subfolders")
        return []
    
    print(f"Found {len(zip_files)} ZIP files:")
    for file in zip_files:
        print(f"  - {file}")
    print("-" * 50)
    
    all_extracted_files = []
    
    for zip_file in zip_files:
        # Extract to the same folder as the ZIP file
        extracted_files = extract_json_from_zip(zip_file, extract_to_same_folder=True)
        all_extracted_files.extend(extracted_files)
    
    print(f"\nSuccessfully extracted {len(all_extracted_files)} JSON files")
    return all_extracted_files

def extract_json_from_zips_by_name(folder_path, name_filter=None):
    """
    Extract JSON files from ZIP files, naming JSON files based on ZIP filenames.
    This is useful when ZIP files were created from individual JSON files.
    Each JSON file is extracted to the same folder as its ZIP file.
    
    Args:
        folder_path (str): Path to the folder to search for ZIP files
        name_filter (str): Only process ZIP files whose names start with this string
        
    Returns:
        list: List of all extracted JSON file paths
    """
    
    # Find all ZIP files with optional name filter
    zip_files = find_zip_files(folder_path, name_filter)
    
    if not zip_files:
        if name_filter:
            print(f"No ZIP files starting with '{name_filter}' found in '{folder_path}' and its subfolders")
        else:
            print(f"No ZIP files found in '{folder_path}' and its subfolders")
        return []
    
    print(f"Found {len(zip_files)} ZIP files:")
    for file in zip_files:
        print(f"  - {file}")
    print("-" * 50)
    
    all_extracted_files = []
    
    for zip_file in zip_files:
        # Extract to the same folder as the ZIP file
        output_folder = zip_file.parent
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zipf:
                file_list = zipf.namelist()
                
                # Use ZIP filename (without .zip) as JSON filename
                json_base_name = zip_file.stem
                
                # If there's only one file in ZIP, extract it
                if len(file_list) == 1:
                    file_name = file_list[0]
                    file_content = zipf.read(file_name).decode('utf-8')
                    
                    if is_json_content(file_content):
                        json_filename = f"{json_base_name}.json"
                        output_file_path = output_folder / json_filename
                        
                        # Overwrite existing file if it exists
                        if output_file_path.exists():
                            print(f"Overwriting existing: {output_file_path}")
                        
                        # Write JSON file with pretty formatting
                        with open(output_file_path, 'w', encoding='utf-8') as f:
                            json_data = json.loads(file_content)
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                        
                        all_extracted_files.append(output_file_path)
                        
                        # Show file sizes
                        zip_size = zip_file.stat().st_size
                        json_size = output_file_path.stat().st_size
                        expansion_ratio = (json_size / zip_size - 1) * 100 if zip_size > 0 else 0
                        
                        print(f"Extracted: {zip_file.name} -> {output_file_path.name}")
                        print(f"  Compressed: {zip_size} bytes -> Uncompressed: {json_size} bytes ({expansion_ratio:.1f}% expansion)")
                    
                    else:
                        print(f"Skipped non-JSON content in: {zip_file.name}")
                
                else:
                    # Multiple files in ZIP - extract each with numbered names
                    for i, file_name in enumerate(file_list):
                        file_content = zipf.read(file_name).decode('utf-8')
                        
                        if is_json_content(file_content):
                            if len(file_list) == 1:
                                json_filename = f"{json_base_name}.json"
                            else:
                                json_filename = f"{json_base_name}_{i+1}.json"
                            
                            output_file_path = output_folder / json_filename
                            
                            # Overwrite existing file if it exists
                            if output_file_path.exists():
                                print(f"Overwriting existing: {output_file_path}")
                            
                            # Write JSON file
                            with open(output_file_path, 'w', encoding='utf-8') as f:
                                json_data = json.loads(file_content)
                                json.dump(json_data, f, indent=2, ensure_ascii=False)
                            
                            all_extracted_files.append(output_file_path)
                            print(f"Extracted: {zip_file.name}:{file_name} -> {output_file_path.name}")
        
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")
    
    print(f"\nSuccessfully extracted {len(all_extracted_files)} JSON files to their original folders")
    return all_extracted_files

def validate_extracted_json_files(json_files):
    """
    Validate that extracted files are valid JSON.
    
    Args:
        json_files (list): List of JSON file paths to validate
    """
    if not json_files:
        print("No JSON files to validate")
        return
    
    print(f"\nValidating {len(json_files)} extracted JSON files:")
    print("-" * 50)
    
    valid_count = 0
    invalid_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json.load(f)
            print(f"✓ Valid: {json_file.name}")
            valid_count += 1
        except Exception as e:
            print(f"✗ Invalid: {json_file.name} - {e}")
            invalid_count += 1
    
    print(f"\nValidation Summary:")
    print(f"Valid JSON files: {valid_count}")
    print(f"Invalid JSON files: {invalid_count}")

def main():
    """Main function using local variables"""
    
    # Configuration - modify these variables as needed
    folder_to_search = "./data"          # Path to folder to search for ZIP files
    name_filter = "instances"            # Only process ZIP files starting with this name
    use_zip_names = True                 # Whether to name JSON files based on ZIP filenames
    validate_json = True                 # Whether to validate extracted JSON files
    
    print(f"Searching for ZIP files in: {folder_to_search}")
    print(f"ZIP name filter: Files starting with '{name_filter}'")
    print(f"Use ZIP names for JSON files: {use_zip_names}")
    print(f"Extract to: Same folder as ZIP files (original locations)")
    print("=" * 60)
    
    # Extract JSON files from ZIP files to their original folders
    if use_zip_names:
        # This method names JSON files based on ZIP filenames (good for reversing the zipping process)
        extracted_files = extract_json_from_zips_by_name(
            folder_path=folder_to_search,
            name_filter=name_filter
        )
    else:
        # This method preserves original filenames from inside ZIP files
        extracted_files = extract_all_json_from_zips(
            folder_path=folder_to_search,
            name_filter=name_filter
        )
    
    # Validate extracted JSON files if requested
    if extracted_files and validate_json:
        validate_extracted_json_files(extracted_files)

def example_usage():
    """Example of how to use the functions"""
    
    # Example 1: Extract only 'instances' ZIP files using ZIP filenames
    print("Example 1: Extract 'instances' ZIP files to original folders")
    extracted1 = extract_json_from_zips_by_name(
        folder_path="./data",
        name_filter="instances"
    )
    
    # Example 2: Extract all ZIP files starting with 'config' 
    print("\nExample 2: Extract 'config' ZIP files")
    extracted2 = extract_all_json_from_zips(
        folder_path="./backup_data",
        name_filter="config"
    )
    
    # Example 3: Extract ZIP files starting with 'temp'
    print("\nExample 3: Extract 'temp' ZIP files")
    extracted3 = extract_json_from_zips_by_name(
        folder_path="./project_files",
        name_filter="temp"
    )
    
    # Validate extracted files
    if extracted1:
        validate_extracted_json_files(extracted1)

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment the line below to run examples instead
    # example_usage()