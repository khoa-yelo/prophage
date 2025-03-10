#!/bin/bash
#SBATCH --job-name=pharokka
#SBATCH --cpus-per-task=1
#SBATCH --partition=nih_s10
#SBATCH --account=mpsnyder
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-49
#SBATCH --output=/labs/mpsnyder/khoa/logs/pharokka/pharokka_%A_%a.out
#SBATCH --error=/labs/mpsnyder/khoa/logs/pharokka/pharokka_%A_%a.err

# Directories
DATA_DIR="/labs/mpsnyder/khoa/data/prophage-DB/bacterial_host_prophages/"
OUTPUT_DIR="/labs/mpsnyder/khoa/data/prophage-DB/annotations/"
DB_DIR="/labs/mpsnyder/khoa/data/pharokka_v1.4.0_databases"
LYSOGEN_FILES="/labs/mpsnyder/khoa/data/prophage-DB/lysogen.txt"
# Use SLURM_ARRAY_TASK_COUNT for the number of parallel jobs
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT}

# Create an array of all FASTA files (assuming extension .fna)
# files=(${DATA_DIR}*.fna)
mapfile -t files < lysogen.txt
total_files=${#files[@]}

echo "Total files: ${total_files}"
echo "Current SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_TASK_COUNT (NUM_JOBS): ${NUM_JOBS}"

# Loop over every file index
for (( i=0; i<total_files; i++ )); do
    # Skip this file if its designated job does not match the current SLURM_ARRAY_TASK_ID
    if [ $(( i % NUM_JOBS )) -ne ${SLURM_ARRAY_TASK_ID} ]; then
        continue
    fi

    input_file=${files[$i]}
    base=$(basename "$input_file" .fna)
    output_path="${OUTPUT_DIR}/pharokka_${base}"

    # Check if the output directory already exists
    if [ -d "$output_path" ]; then
        echo "Skipping $input_file - Output directory exists: $output_path"
        continue
    fi

    echo "Processing file index $i: $input_file"

    pharokka.py -i "$input_file" \
                -o "$output_path" \
                -d "$DB_DIR" \
                -t 8 \
                -p "$base" \
                -f \
                --fast
done
