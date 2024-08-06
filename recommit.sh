#!/bin/bash

# Number of files to commit per batch
BATCH_SIZE=10

# Counter to track the number of files committed
COUNT=0

# Array to store files for each batch
FILE_BATCH=()

# Iterate over each file in the working directory
for FILE in $(git diff --name-only); do
  FILE_BATCH+=("$FILE")
  COUNT=$((COUNT + 1))

  # If the batch size is reached, commit the batch
  if [ $COUNT -ge $BATCH_SIZE ]; then
    git add "${FILE_BATCH[@]}"
    git commit -m "Batch commit"
    FILE_BATCH=()
    COUNT=0
  fi
done

# Commit any remaining files
if [ ${#FILE_BATCH[@]} -gt 0 ]; then
  git add "${FILE_BATCH[@]}"
  git commit -m "Final batch commit"
fi

# Push the commits
git push origin main

