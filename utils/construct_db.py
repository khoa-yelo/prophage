import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import os
from os.path import join
from glob import glob


virus_path = "/orange/sai.zhang/khoa/repos/prophage/data/virome_db.csv"
genome_path = "/orange/sai.zhang/khoa/repos/prophage/data/genome_db.csv"

df = pd.read_csv(genome_path)
df_virus = pd.read_csv(virus_path)

df_protein_name = pd.read_csv("/orange/sai.zhang/khoa/repos/prophage/data/protein_names.csv")
protein_name_map = df_protein_name.dropna().drop_duplicates("protein_id").\
                                            set_index("protein_id")["protein"].dropna()
df["product"] = df.protein_id.map(protein_name_map)

def is_in_prophage(row):
    # grab all prophage regions for this record_id
    regs = df_virus[df_virus['record_id'] == row['record_id']]
    # check if any region fully covers this feature
    return int(
        ((regs['start'] <= row['start']) & (regs['end'] >= row['end'])).any()
    )

# # apply and create the new column
df['prophage'] = df.apply(is_in_prophage, axis=1)
# save the updated DataFrame to a new CSV file
new_genome_path = "/orange/sai.zhang/khoa/repos/prophage/data/genome_db_new.csv"

df.to_csv(new_genome_path, index=False)
