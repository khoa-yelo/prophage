import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Select taxa based on assembly length.")
    parser.add_argument("--input", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, help="Path to the output text file.")
    return parser.parse_args()


def main():
    
    args = parse_args()
    path = args.input
    output = args.output

    with open(path, "r") as f:
        metadata = f.readlines()
    # Dictionary to hold the longest Nanopore sequence per species
    species_best = {}
    for line in metadata:
        entry = json.loads(line)
        tech = entry.get("assembly_info", {}).get("sequencing_tech", "")
        if not tech or "nano" not in tech.lower() or "illumina" in tech.lower():
            continue  # skip if not Nanopore
        tax_id = entry["organism"]["tax_id"]
        length = entry["assembly_stats"]["total_ungapped_length"]
        # If species not seen or current sequence is longer, update it
        if tax_id not in species_best or length > species_best[tax_id]["assembly_stats"]["total_ungapped_length"]:
            species_best[tax_id] = entry
    # Extract results
    selected = list(species_best.values())
    orgs = [entry["organism"]["tax_id"] for entry in selected]
    stats = [int(entry["assembly_stats"]["total_ungapped_length"]) for entry in selected]
    accessions = [access["accession"] for access in selected]

    with open(output, "w") as f:
        for acc in accessions:
            f.write(acc)
            f.write("\n")

if __name__ == "__main__":
    main()