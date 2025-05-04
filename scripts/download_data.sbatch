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


datasets download genome accession \
  --inputfile selected_bacteria.txt \
  --include genome,gff3,gtf,cds,protein,rna \
  --filename bacterial_genomes.zip