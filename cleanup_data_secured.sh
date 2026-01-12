#!/bin/bash

# Script to delete folders and JSON files in specific subdirectories
# Location: /securedstorage/DATAsec/cole/Data-secured (Copy)/

BASE_DIR="/securedstorage/DATAsec/cole/Data-secured (Copy)"

# Array of folders to process
FOLDERS=(
    "Benz_control"
    "EB_control"
    "hex_control"
    "opto_3-oct"
    "opto_ACV"
    "opto_AIR"
    "opto_benz"
    "opto_benz_1"
    "opto_EB"
    "opto_EB(6-training)"
    "opto_hex"
)

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    exit 1
fi

cd "$BASE_DIR" || exit 1

echo "=== Data Cleanup Script ==="
echo "Base directory: $BASE_DIR"
echo ""
echo "This script will delete all subdirectories and JSON files in:"
for folder in "${FOLDERS[@]}"; do
    echo "  - $folder"
done
echo ""

# Ask for confirmation
read -p "Are you sure you want to proceed? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Operation cancelled."
    exit 0
fi

# Process each folder
for folder in "${FOLDERS[@]}"; do
    folder_path="$BASE_DIR/$folder"
    
    if [ ! -d "$folder_path" ]; then
        echo "Warning: Folder not found: $folder_path"
        continue
    fi
    
    echo ""
    echo "Processing: $folder"
    
    # Process each fly folder (subdirectory)
    for fly_folder in "$folder_path"/*; do
        if [ -d "$fly_folder" ]; then
            fly_name=$(basename "$fly_folder")
            
            # Count items inside the fly folder
            subdirs=$(find "$fly_folder" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            jsonfiles=$(find "$fly_folder" -mindepth 1 -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
            
            if [ "$subdirs" -gt 0 ] || [ "$jsonfiles" -gt 0 ]; then
                echo "  Processing fly folder: $fly_name"
                
                # Delete subdirectories inside fly folder
                if [ "$subdirs" -gt 0 ]; then
                    find "$fly_folder" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
                    echo "    ✓ Deleted $subdirs subdirectories"
                fi
                
                # Delete JSON files inside fly folder
                if [ "$jsonfiles" -gt 0 ]; then
                    find "$fly_folder" -mindepth 1 -maxdepth 1 -name "*.json" -delete
                    echo "    ✓ Deleted $jsonfiles JSON files"
                fi
            fi
        fi
    done
done

echo ""
echo "=== Cleanup Complete ==="
