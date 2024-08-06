#!/bin/bash

# Number of files to commit per batch
BATCH_SIZE=10

# Counter to track the number of files committed
COUNT=0

# Array to store files for each batch
FILE_BATCH=()

# Function to commit and push a batch of files
commit_and_push_batch() {
  if [ ${#FILE_BATCH[@]} -gt 0 ]; then
    echo "Staging files: ${FILE_BATCH[@]}"
    git add "${FILE_BATCH[@]}"
    git commit -m "Batch commit of ${#FILE_BATCH[@]} files"
    echo "Pushing batch..."
    git push origin main
    if [ $? -ne 0 ]; then
      echo "Push failed, retrying in 30 seconds..."
      sleep 30
      git push origin main
    fi
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
    echo "Pushed a batch of $BATCH_SIZE files. Pausing for 30 seconds..."
    sleep 30
  fi
done

# Commit and push any remaining files
commit_and_push_batch

echo "All batches pushed."

