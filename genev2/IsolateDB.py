import sqlite3
import pandas as pd
from typing import Optional, List, Union, Dict

class IsolateDB:
    def __init__(self, db_path: str):
        """
        Open a connection to the SQLite database at db_path.
        """
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def get_viral_element(
        self,
        species: Optional[str] = None,
        columns: str = 'protein_id'
    ) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
        """
        Return viral elements grouped by virome.seq_name.

        Args:
            species: If provided, filter virome entries by this species.
            columns: 'protein_id' to return only protein IDs;
                     'all' to return lists of type, biotype, protein_id, homologs, and product.

        Returns:
            If columns == 'protein_id':
                Dict[seq_name, List[protein_id]]
            If columns == 'all':
                Dict[seq_name, Dict[str, List[str]]], keys 'type', 'biotype', 'protein_id', 'homologs', 'product'.
        """
        if columns not in ('protein_id', 'all'):
            raise ValueError("columns must be 'protein_id' or 'all'")

        if columns == 'protein_id':
            select_cols = "g.protein_id"
        else:
            select_cols = "g.type, g.biotype, g.protein_id, g.homologs, g.product"

        query = f"""
        SELECT v.seq_name, {select_cols}
        FROM virome AS v
        JOIN genome AS g
          ON v.record_id = g.record_id
         AND g.prophage <> 0
         AND g.start >= v.start
         AND g.end   <= v.end
        """
        params: tuple = ()
        if species is not None:
            query += " WHERE v.species = ?"
            params = (species,)

        df = pd.read_sql_query(query, self.conn, params=params)

        if columns == 'protein_id':
            return df.groupby('seq_name')['protein_id']\
                     .apply(list)\
                     .to_dict()
        else:
            grouped: Dict[str, Dict[str, List[str]]] = {}
            for name, group in df.groupby('seq_name'):
                grouped[name] = {
                    'type': group['type'].tolist(),
                    'biotype': group['biotype'].tolist(),
                    'protein_id': group['protein_id'].tolist(),
                    'homologs': group['homologs'].tolist(),
                    'product': group['product'].tolist()
                }
            return grouped

    def get_species(self, species: str) -> pd.DataFrame:
        """
        Return the entire genome table filtered to the given species.
        """
        query = "SELECT * FROM genome WHERE species = ?"
        return pd.read_sql_query(query, self.conn, params=(species,))

    def get_product(self, protein_ids: List[str]) -> pd.DataFrame:
        """
        Given a list of protein_ids, return a DataFrame with one row per protein_id,
        picking the first matching product.
        """
        if not protein_ids:
            return pd.DataFrame(columns=["protein_id", "product"])

        placeholders = ",".join("?" for _ in protein_ids)
        query = f"""
        SELECT protein_id, product
          FROM genome
         WHERE protein_id IN ({placeholders})
        """
        df = pd.read_sql_query(query, self.conn, params=protein_ids)
        return df.drop_duplicates(subset=['protein_id'], keep='first')

    def get_taxonomy(
        self,
        species: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Return a mapping from virome.seq_name to taxonomy.
        If species is provided, filters by species.
        """
        query = "SELECT seq_name, taxonomy FROM virome"
        params: tuple = ()
        if species is not None:
            query += " WHERE species = ?"
            params = (species,)

        df = pd.read_sql_query(query, self.conn, params=params)
        return df.set_index('seq_name')['taxonomy'].to_dict()
   
    def get_virome_table(self) -> pd.DataFrame:
        """
        Return the entire virome table as a DataFrame.
        """
        query = "SELECT * FROM virome"
        return pd.read_sql_query(query, self.conn)
    
    def get_genome_table(self) -> pd.DataFrame:
        """
        Return the entire genome table as a DataFrame.
        """
        query = "SELECT * FROM genome"
        return pd.read_sql_query(query, self.conn)
    
    def get_all_record_ids(self) -> List[Union[int, str]]:
        """
        Return a list of all record_id values from the genome table.
        """
        query = "SELECT DISTINCT record_id FROM genome"
        df = pd.read_sql_query(query, self.conn)
        return df['record_id'].tolist()

    def get_record_feature(self, record_id: Union[str,int]) -> Dict[str, Union[str,int]]:
        """
        Return features for a given record_id from the genome table.
        Keys: species, type, biotype, protein_id, prophage, homologs
        """
        query = """
        SELECT species, record_id, type, biotype, protein_id, prophage, homologs, locus_tag, strand, species_prev, product
          FROM genome
         WHERE record_id = ?
        """
        df = pd.read_sql_query(query, self.conn, params=(record_id,))
        if df.empty:
            return {}

        df['species_prev'] = df['species_prev'].fillna(0.)
        df['homologs'] = df['homologs'].fillna(0.) 
        df['product'] = df['product'].fillna("") 
    
        return {
            'species': df['species'].values,
            'records': df['record_id'].values,
            'type': df['type'].values,
            'biotype': df['biotype'].values,
            'protein_id': df['protein_id'].values,
            'prophage': df['prophage'].values,
            'homologs': df['homologs'].values,
            'locus_tag': df['locus_tag'].values,
            'strand': df['strand'].values,
            'species_prev': df['species_prev'].values,
            'product': df['product'].values,

        }
    
    def get_all_types(self) -> List[str]:
        """
        Return a list of all distinct types in the genome table.
        """
        query = "SELECT DISTINCT type FROM genome"
        df = pd.read_sql_query(query, self.conn)
        return df['type'].tolist()

    def get_all_biotypes(self) -> List[str]:
        """
        Return a list of all distinct biotypes in the genome table.
        """
        query = "SELECT DISTINCT biotype FROM genome"
        df = pd.read_sql_query(query, self.conn)
        return df['biotype'].tolist()
    
    def close(self):
        """
        Close the underlying database connection.
        """
        self.conn.close()
