#!/bin/bash
#SBATCH --job-name=download-ncbi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=interactive
#SBATCH --account=default
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --mem=8GB

### 
### Load the required modules
eval "$($SCRATCH/envs/micromamba/bin/micromamba shell hook --shell=bash)"
micromamba activate ncbi_dataset

###
### Download NCBI datasets
input_file=$1
output_file=$2
datasets download genome accession \
  --inputfile $input_file \
  --include genome,gff3,gtf,cds,protein,rna \
  --filename $output_file 