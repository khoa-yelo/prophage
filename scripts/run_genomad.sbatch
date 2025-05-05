#!/bin/bash
#SBATCH --job-name=genomad
#SBATCH --cpus-per-task=1
#SBATCH --partition=nih_s10
#SBATCH --account=mpsnyder
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --array=0-100
#SBATCH --output=logs/genomad2/genomad_%A_%a.out
#SBATCH --error=logs/genomad2/genomad_%A_%a.err
eval "$($SCRATCH/envs/micromamba/bin/micromamba shell hook --shell=bash)"
micromamba activate genomad
# Directories
#DATA_DIR="$SCRATCH/data/prophage-DB/bacterial_host_prophages/"
OUTPUT_DIR="$SCRATCH/data/virome/genomad_outs" #"$SCRATCH/data/prophage-DB/annotations_genomad_PROGS_oneint/"
DB_DIR="$SCRATCH/data/genomad_db"
FASTA_FILES="$SCRATCH/data/virome/ncbi_dataset/genomic_fna_list.txt" #"$SCRATCH/data/prophage-DB/int_high_quality_complete_sample800.txt"
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
    output_path="${OUTPUT_DIR}/genomad_${base}"

    # Check if the output directory already exists
    if [ -d "$output_path" ]; then
        echo "Skipping $input_file - Output directory exists: $output_path"
        continue
    fi

    echo "Processing file index $i: $input_file"

    # genomad annotate \
    #      "$input_file" \
    #      "$output_path" \
    #      "$DB_DIR" 

    genomad end-to-end \
            "$input_file" \
            "$output_path" \
            "$DB_DIR" \
            --relaxed \
            --threads 8 \

done
