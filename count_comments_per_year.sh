#!/bin/bash




# Specify the directory containing your JSON files
dir_path="/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskMen_download/AskMen_per_year_comments_14M_adjusted"

# Initialize an associative array to hold the count of comments per year
declare -A comments_per_year

# Initialize a counter for processed files
processed_files=0

# Pre-calculate the total number of files to process
total_files=$(ls -1 $dir_path/*.json 2>/dev/null | wc -l)

# Iterate over JSON files that match the pattern
for file_name in $dir_path/*.json; do
  # Use jq to parse JSON and extract the created_utc value, then convert it to a date string, and then extract the year
  while IFS= read -r year; do
    ((comments_per_year[$year]++))
  done < <(jq -r '(.created_utc | tonumber | todate | strptime("%Y-%m-%dT%H:%M:%SZ") | .[0])' $file_name)
  
  # Increment and print the number of processed files
  ((processed_files++))
  echo "Processed files: $processed_files/$total_files"
  
  # Display intermediate results after each file
  for year in "${!comments_per_year[@]}"; do
    echo "Current count for $year: ${comments_per_year[$year]}"
  done
done

# Print the final number of comments per year
echo "Final count of comments per year:"
for year in "${!comments_per_year[@]}"; do
  echo "$year: ${comments_per_year[$year]}"
done