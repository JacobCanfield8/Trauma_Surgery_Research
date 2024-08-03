#!/bin/bash

count=1

# Function to rename files in a directory
rename_files() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively rename files in it
            rename_files "$file"
        elif [ -f "$file" ]; then
            # If it's a file, rename it unless it has a .md extension
            if [[ "$file" != *.md ]]; then
                dir=$(dirname "$file")
                ext="${file##*.}"
                new_name="${dir}/file$(printf "%04d" $count).${ext}"
                mv "$file" "$new_name"
                count=$((count + 1))
            fi
        fi
    done
}

# Start renaming files from the current directory
rename_files "."

echo "Renamed $count files."

