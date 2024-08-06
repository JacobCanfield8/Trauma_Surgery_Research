#!/bin/bash

# Number of files to commit per batch
BATCH_SIZE=100

# Counter to track the number of files committed
COUNT=0

# Array to store files for each batch
FILE_BATCH=()

# Function to commit and push a batch of files
commit_and_push_batch() {
  if [ ${#FILE_BATCH[@]} -gt 0 ]; then
    git add "${FILE_BATCH[@]}"
    git commit -m "Batch commit"
    git push origin main
    FILE_BATCH=()
    COUNT=0
  fi
}

# Iterate over modified and untracked files
for FILE in $(git status --porcelain | awk '{print $2}'); do
  FILE_BATCH+=("$FILE")
  COUNT=$((COUNT + 1))

  # If the batch size is reached, commit and push the batch
  if [ $COUNT -ge $BATCH_SIZE ]; then
    commit_and_push_batch
  fi
done

# Commit and push any remaining files
commit_and_push_batch

