#this code is to check files that are not in the reference list and move them to a unique_downloads folder

import os
import shutil
import re

def normalize_filename(filename):
    # Remove file extension and _transcript suffix
    filename = os.path.splitext(filename)[0]
    filename = re.sub(r'_transcript$', '', filename)
    
    # Remove special characters, spaces, and convert to lowercase
    normalized = re.sub(r'[^a-zA-Z0-9]', '', filename.lower())
    
    # Remove "lexfridmanpodcast" and any following numbers
    normalized = re.sub(r'lexfridmanpodcast\d+', '', normalized)
    
    return normalized

def get_normalized_files(folder_path, file_extension):
    return {normalize_filename(f): f for f in os.listdir(folder_path) if f.lower().endswith(file_extension)}

def move_file(src_folder, dst_folder, filename):
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, filename)
    shutil.move(src_path, dst_path)
    print(f"Moved: {filename}")

def compare_and_move_files(lex_diarized_path, downloads_path, unique_downloads_path, reference_list=None):
    # Create the unique_downloads folder if it doesn't exist
    os.makedirs(unique_downloads_path, exist_ok=True)

    # Get normalized filenames from both folders
    lex_diarized_files = get_normalized_files(lex_diarized_path, '.txt')
    download_files = get_normalized_files(downloads_path, '.mp3')

    # Normalize the reference list if provided
    if reference_list:
        normalized_reference = set(normalize_filename(item) for item in reference_list)
    else:
        normalized_reference = set()

    moved_count = 0

    # Compare files and move non-matching ones
    for norm_filename, original_filename in download_files.items():
        # Check if the file is not in lex_diarized folder
        if norm_filename not in lex_diarized_files:
            # Check if the file is not in the reference list (if provided)
            if not normalized_reference or norm_filename not in normalized_reference:
                move_file(downloads_path, unique_downloads_path, original_filename)
                moved_count += 1

    # Check for files in reference list that are not in lex_diarized
    for norm_ref in normalized_reference:
        if norm_ref not in lex_diarized_files:
            # Check if the file exists in downloads
            matching_download = next((f for n, f in download_files.items() if n == norm_ref), None)
            if matching_download:
                move_file(downloads_path, unique_downloads_path, matching_download)
                moved_count += 1

    print(f"Moved {moved_count} non-matching mp3 files to {unique_downloads_path}")

# Example usage
lex_diarized_path = "/podAI/podAI/lex_diarized"
downloads_path = "/podAI/podAI/downloads"
unique_downloads_path = "/podAI/podAI/unique_downloads"

compare_and_move_files(lex_diarized_path, downloads_path, unique_downloads_path)
