# prophage

Finding new viruses from long reads isolates

## Overview
WIP

## Repo Structure
WIP

## Setup
WIP
```

```


### Download Raw Data from NCBI


Download metadata from NCBI
```
sbatch scripts/download_metadata.sbatch bacteria.json

```

Select for taxa of interest

```
python3 select_taxa.py --input bacteria.json --output selected_bacteria.txt
```

Download sequence and annotate all sequences

```
sbatch scripts/download_data.sbatch selected_bacteria.txt bacterial_genomes.zip
```

Total 1958 nanopore isolates assembled sequences 
Total 8,097,917 proteins saved as `all_proteins.fasta`

### Identify virus with geNomad

```
sbatch scripts/run_genomad.sbatch
```

### Protein Clustering

Initial cluster based on sequence similarity with MMSeqs2

```
sbatch --wrap "mmseqs createdb all_proteins.faa bacterial_prot_mmseqs_db" -t 24:00:00 --mem 24Gb -p interactive


sbatch --wrap "mmseqs cluster bacterial_prot_mmseqs_db clusterRes_id03_c08 tmp3 --min-seq-id 0.3 -c 0.8" -t 24:00:00 --mem 40Gb -p interactive


mmseqs createtsv bacterial_prot_mmseqs_db bacterial_prot_mmseqs_db linClusterRes_id03_c08 clusters_linclust_id03_c08.tsv

```

Obtain a list of representative unique proteins, 1,063,267 sequences, need to embed them

```
cut -f1 clusters_mmseqclust_id03_c08.tsv | sort | uniq > unique_proteins.txt
seqkit grep -f unique_proteins.txt all_proteins_dedup.faa > unique_proteins.faa
```

### Protein Embeddings

### Curated Dataset


### Models

