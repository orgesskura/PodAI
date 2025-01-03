# This script removes the full transcript from the diarized files and only keeps the diarized utterances.

import os
import shutil

input_folder = "temp"
output_folder = "cleaned_diarized"

# Make sure the output folder exists, create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        # Read the file
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        # Check if the file has the specific structure
        utterances = []
        in_conversation = False
        for line in lines:
            if "Utterances:" in line:
                in_conversation = True
                continue
            if in_conversation:
                utterances.append(line)

        # If the file was structured with utterances, save the cleaned version
        if utterances:
            with open(output_file_path, "w") as file:
                file.writelines(utterances)
        else:
            # If the file does not match the structure, just copy it over
            shutil.copy(input_file_path, output_file_path)

print("Processing complete. Cleaned files are in:", output_folder)
