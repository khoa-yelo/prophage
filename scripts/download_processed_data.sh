#!/bin/bash
# Download processed data from Google Cloud Storage

output_dir=$1
if [ -z "$output_dir" ]; then
  echo "Usage: $0 <output_directory>"
  exit 1
fi
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi
# Download the processed data
gsutil -m cp -r gs://bacterial-isolates $output_dir