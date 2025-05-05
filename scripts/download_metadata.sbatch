#!/bin/bash
#SBATCH --job-name=download-meta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=interactive
#SBATCH --account=default
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --mem=8GB

###
# Load the required modules
eval "$($SCRATCH/envs/micromamba/bin/micromamba shell hook --shell=bash)"
micromamba activate ncbi_dataset

output_file=$1
# Download the metadata for all complete bacterial genomes released between 2015 and 2025
datasets summary genome taxon eubacteria --assembly-level complete --annotated --as-json-lines --released-after 2015-01-01 --released-before 2025-01-01 > $output_file