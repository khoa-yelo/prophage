#!/bin/bash
#SBATCH --job-name=dna-feat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=nih_s10
#SBATCH --account=mpsnyder
#SBATCH --time=3:00:00
#SBATCH --mem=3GB
#SBATCH --array=0-9               # now 10 tasks: 0–9
#SBATCH --output=logs/%A_%a.out   # separate log per array index

# make sure our log directory exists
mkdir -p logs

for i in {1..100}; do
    if (( i % 10 == SLURM_ARRAY_TASK_ID )); then
        # pad to three digits
        idx=$(printf "%03d" "$i")
        python3 dnafeat.py \
            "/labs/mpsnyder/khoa/repos/prophage/data/split_cds/all_cds.part_${idx}.fasta" \
            --kmer_k     4 \
            --pse_lambda 5 \
            --pse_w      0.1 \
            --cgr_res    4 \
            -o "/labs/mpsnyder/khoa/repos/prophage/data/split_cds/all_cds.part_${idx}.features.csv"
    fi
done