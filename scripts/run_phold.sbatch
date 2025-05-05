#!/bin/bash
#SBATCH --job-name=phold
#SBATCH --cpus-per-task=1
#SBATCH --partition=interactive
#SBATCH --account=default
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-10
#SBATCH --output=logs/phold/phold_%A_%a.out
#SBATCH --error=logs/phold/phold_%A_%a.err

eval "$($SCRATCH/envs/micromamba/bin/micromamba shell hook --shell=bash)"
micromamba activate phold
# Directories
DATA_DIR="$SCRATCH/data/prophage-DB/bacterial_host_prophages/"
OUTPUT_DIR="$SCRATCH/data/prophage-DB/annotations_phold_GENOMAD_noint"
# DB_DIR="$SCRATCH/data/genomad_db"
FASTA_FILES="$SCRATCH/data/prophage-DB/no_int_high_quality_complete_genomad.txt"
# Use SLURM_ARRAY_TASK_COUNT for the number of parallel jobs
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT}

# Create an array of all FASTA files (assuming extension .fna)
# files=(${DATA_DIR}*.fna)
mapfile -t files < $FASTA_FILES
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
    output_path="${OUTPUT_DIR}/phold_${base}"

    # Check if the output directory already exists
    if [ -d "$output_path" ]; then
        echo "Skipping $input_file - Output directory exists: $output_path"
        continue
    fi

    echo "Processing file index $i: $input_file"

    phold run \
    -i $input_file \
    -o $output_path \
    -p phold_  \
    -f \
    --cpu
done
