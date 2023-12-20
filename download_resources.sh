#!/bin/bash

# Check if gdown is installed, if not, install it
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Set the source folder name and ID on Google Drive
source_folder_name="Resources"
source_folder_id="1ikOYp_37YUczyncHvmSpVoL4PQuz7zkm"

# Download the entire folder from Google Drive
gdown --folder https://drive.google.com/drive/folders/${source_folder_id} --remaining-ok

echo "Resources downloaded successfully."

# Extract datasets and checkpoints from the downloaded folder
for subfolder in $(ls "${source_folder_name}"); do
    mv ${source_folder_name}/${subfolder} .
done

# Remove the parent folder
rm -r "${source_folder_name}"

