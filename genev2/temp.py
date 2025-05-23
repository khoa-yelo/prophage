class IsolateData:
    def __init__(self, virus_db_path, genome_db_path):
        self.virus_db = pd.read_csv(virus_db_path)
        self.genome_db = pd.read_csv(genome_db_path)
        self.species_list = list(self.virus_db['species'].unique())

    def get_all_species(self):
        return self.species_list

    def get_proteins_from_species(self, species):
        if species not in self.species_list:
            raise ValueError(f"Species '{species}' not found in virus DB")
        return {
            'viral_proteins': self.get_viral_proteins_from_species(species),
            'host_proteins': self.get_host_proteins_from_species(species)
        }
    
    def _get_protein_coding_genes(self, species):
        df = self.genome_db
        mask = (
            (df['species'] == species) &
            (df['biotype'] == 'protein_coding') &
            (df['protein_id'].notnull())
        )
        return df[mask].copy()

    def _get_viral_regions(self, species):
        df = self.virus_db[self.virus_db['species'] == species].copy()
        df[['start', 'end']] = df['coordinates'].str.split('-', expand=True).astype(int)
        return df[['seq_name', 'record_id', 'start', 'end']]

    def _get_contig_lengths(self, species):
        df = self.genome_db[self.genome_db['species'] == species]
        return df.groupby('record_id')['end'].max().to_dict()

    def _merge_intervals(self, intervals):
        merged = []
        for start, end in sorted(intervals, key=lambda x: x[0]):
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged

    def _compute_host_regions(self, species):
        """
        For all contigs in the genome (record_id), subtract any viral intervals
        (merged per contig) from [1..contig_end] to yield host regions.
        Includes contigs with no viral regions.
        Returns DataFrame with ['seq_name', 'record_id', 'start', 'end'].
        """
        viral_regions = self._get_viral_regions(species)
        contig_ends = self._get_contig_lengths(species)
        host_records = []

        # Iterate over all contigs in genome
        for contig_id, max_end in contig_ends.items():
            # Get viral intervals on this contig, if any
            contig_viral = viral_regions[viral_regions['record_id'] == contig_id]
            if not contig_viral.empty:
                intervals = contig_viral[['start', 'end']].values.tolist()
                merged_intervals = self._merge_intervals(intervals)
            else:
                merged_intervals = []

            pointer = 1
            region_idx = 1
            # Carve host segments around merged viral spans
            for v_start, v_end in merged_intervals:
                if pointer < v_start:
                    host_records.append({
                        'seq_name': f"{contig_id}_host{region_idx}",
                        'record_id': contig_id,
                        'start': pointer,
                        'end': v_start - 1
                    })
                    region_idx += 1
                pointer = v_end + 1

            # Tail region (covers entire contig if no viral spans)
            if pointer <= max_end:
                host_records.append({
                    'seq_name': f"{contig_id}_host{region_idx}",
                    'record_id': contig_id,
                    'start': pointer,
                    'end': max_end
                })

        return pd.DataFrame(host_records)

    def get_viral_proteins_from_species(self, species):
        genes = self._get_protein_coding_genes(species)
        viral_regions = self._get_viral_regions(species)
        result = []
        for _, region in viral_regions.iterrows():
            sel = genes[
                (genes['record_id'] == region['record_id']) &
                (genes['start'] >= region['start']) &
                (genes['end'] <= region['end'])
            ]
            result.append({
                'seq_name': region['seq_name'],
                'record_id': region['record_id'],
                'protein_ids': sel['protein_id'].tolist()
            })
        return result

    def get_host_proteins_from_species(self, species):
        genes = self._get_protein_coding_genes(species)
        host_regions = self._compute_host_regions(species)
        result = []
        for _, region in host_regions.iterrows():
            sel = genes[
                (genes['record_id'] == region['record_id']) &
                (genes['start'] >= region['start']) &
                (genes['end'] <= region['end'])
            ]
            result.append({
                'seq_name': region['seq_name'],
                'record_id': region['record_id'],
                'protein_ids': sel['protein_id'].tolist()
            })
        return result
import pandas as pd

class IsolateData:
    def __init__(self, virus_db_path, genome_db_path):
        self.virus_db = pd.read_csv(virus_db_path)
        self.genome_db = pd.read_csv(genome_db_path)
        self.species_list = list(self.virus_db['species'].unique())

    def get_all_species(self):
        return self.species_list

    def get_proteins_from_species(self, species):
        if species not in self.species_list:
            raise ValueError(f"Species '{species}' not found in virus DB")
        return {
            'viral_proteins': self.get_viral_proteins_from_species(species),
            'host_proteins': self.get_host_proteins_from_species(species)
        }

    def _get_genes_by_biotype(self, species, biotype=None):
        df = self.genome_db
        mask = (df['species'] == species)
        if biotype is None:
            mask &= df['biotype'].notna()
        else:
            mask &= (df['biotype'] == biotype)
            # only protein_coding requires protein_id
            if biotype == 'protein_coding' and 'protein_id' in df.columns:
                mask &= df['protein_id'].notnull()
        cols = df.columns.tolist()
        return df.loc[mask, cols]

    def _get_viral_regions(self, species):
        df = self.virus_db
        mask = df['species'] == species
        return df.loc[mask, ['seq_name', 'record_id', 'start', 'end']]

    def _get_contig_lengths(self, species):
        df = self.genome_db
        mask = df['species'] == species
        return df.loc[mask].groupby('record_id')['end'].max().to_dict()

    def _merge_intervals(self, intervals):
        merged = []
        for start, end in sorted(intervals, key=lambda x: x[0]):
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged

    def _compute_host_regions(self, species):
        viral_regions = self._get_viral_regions(species)
        contig_ends = self._get_contig_lengths(species)
        host_records = []

        for contig_id, max_end in contig_ends.items():
            contig_viral = viral_regions[viral_regions['record_id'] == contig_id]
            intervals = contig_viral[['start', 'end']].values.tolist() if not contig_viral.empty else []
            merged = self._merge_intervals(intervals)

            pointer, region_idx = 1, 1
            for v_start, v_end in merged:
                if pointer < v_start:
                    host_records.append({
                        'seq_name': f"{contig_id}_host{region_idx}",
                        'record_id': contig_id,
                        'start': pointer,
                        'end': v_start - 1
                    })
                    region_idx += 1
                pointer = v_end + 1

            if pointer <= max_end:
                host_records.append({
                    'seq_name': f"{contig_id}_host{region_idx}",
                    'record_id': contig_id,
                    'start': pointer,
                    'end': max_end
                })

        return pd.DataFrame(host_records)

    def get_viral_proteins_from_species(self, species):
        genes = self._get_genes_by_biotype(species, biotype='protein_coding')
        viral_regions = self._get_viral_regions(species)
        result = []
        for _, region in viral_regions.iterrows():
            sel = genes[
                (genes['record_id'] == region['record_id']) &
                (genes['start'] >= region['start']) &
                (genes['end'] <= region['end'])
            ]
            result.append({
                'seq_name': region['seq_name'],
                'record_id': region['record_id'],
                'protein_ids': sel['protein_id'].tolist()
            })
        return result

    def get_host_proteins_from_species(self, species):
        genes = self._get_genes_by_biotype(species, biotype='protein_coding')
        host_regions = self._compute_host_regions(species)
        result = []
        for _, region in host_regions.iterrows():
            sel = genes[
                (genes['record_id'] == region['record_id']) &
                (genes['start'] >= region['start']) &
                (genes['end'] <= region['end'])
            ]
            result.append({
                'seq_name': region['seq_name'],
                'record_id': region['record_id'],
                'protein_ids': sel['protein_id'].tolist()
            })
        return result

    def get_elements_from_species(self, species, biotype=None):
        """
        Return all genome elements for a species.
        If biotype is None, returns all elements where biotype is not null.
        """
        df = self._get_genes_by_biotype(species, biotype=biotype)
        return df.to_dict(orient='records')

    def get_elements_by_biotype(self, biotype):
        """
        Return all elements across all species matching a specific biotype.
        Only filter on protein_id if biotype is 'protein_coding'.
        """
        df = self.genome_db
        mask = (df['biotype'] == biotype)
        if biotype == 'protein_coding' and 'protein_id' in df.columns:
            mask &= df['protein_id'].notnull()
        return df.loc[mask].to_dict(orient='records')
