#!/bin/bash
#SBATCH --job-name=svim
#SBATCH --cpus-per-task=1
#SBATCH --partition=interactive
#SBATCH --account=default
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --output=logs/svim/svim_%A_%a.out
#SBATCH --error=logs/svim/svim_%A_%a.err

module load minimap2
module load samtools

export ref="$SCRATCH/data/sra/reference.fasta"
export assembly="$SCRATCH/data/sra/assembly_contig1.fasta"
export aligned="$SCRATCH/data/sra/alignments.sam"
export aligned_sorted_bam="$SCRATCH/data/sra/alignments.sorted.bam"
export output_dir="$SCRATCH/data/sra/svim_out"

minimap2 -a -x asm5 --cs -r2k -t 16 $ref $assembly > $aligned
samtools sort -m4G -@4 -o $aligned_sorted_bam $aligned
samtools index $aligned_sorted_bam
svim-asm haploid $output_dir $aligned_sorted_bam $ref