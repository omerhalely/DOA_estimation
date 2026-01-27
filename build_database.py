import os
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm


def combine_flac_files(source_path: Union[str, Path], target_folder: Union[str, Path]) -> None:
    """
    Recursively find all .flac files in the source path and copy them to a single target folder.
    
    Args:
        source_path: Initial path inside the data folder to search for .flac files
        target_folder: Destination folder where all .flac files will be copied
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If source_path does not exist
        PermissionError: If there's no permission to read source or write to target
    """
    # Convert to Path objects for easier manipulation
    source_path = Path(source_path)
    target_folder = Path(target_folder)
    
    # Validate source path exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    # Create target folder if it doesn't exist
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all .flac files recursively
    flac_files = list(source_path.rglob("*.flac"))
    
    if not flac_files:
        print(f"No .flac files found in {source_path}")
        return
    
    print(f"Found {len(flac_files)} .flac files")
    
    # Copy each file to the target folder
    copied_count = 0
    skipped_count = 0
    
    for flac_file in tqdm(flac_files, desc="Copying .flac files", unit="file"):
        target_file = target_folder / flac_file.name
        
        # Handle duplicate filenames by appending a counter
        if target_file.exists():
            base_name = flac_file.stem
            extension = flac_file.suffix
            counter = 1
            while target_file.exists():
                target_file = target_folder / f"{base_name}_{counter}{extension}"
                counter += 1
            tqdm.write(f"Duplicate filename detected, renaming to: {target_file.name}")
        
        try:
            shutil.copy2(flac_file, target_file)
            copied_count += 1
        except Exception as e:
            tqdm.write(f"Error copying {flac_file}: {e}")
            skipped_count += 1
    
    print(f"\nCompleted!")
    print(f"Successfully copied: {copied_count} files")
    if skipped_count > 0:
        print(f"Skipped (errors): {skipped_count} files")
    print(f"Target folder: {target_folder.absolute()}")


if __name__ == "__main__":
    # Example usage
    # Adjust these paths according to your needs
    data_folder = os.path.join(os.getcwd(), "data", "train-clean-100")
    output_folder = os.path.join(os.getcwd(), "data", "combined-train-clean-100")
    
    combine_flac_files(data_folder, output_folder)
