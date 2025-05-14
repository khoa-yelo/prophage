import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import StandardScaler
import pandas as pd
import h5py


df = dd.read_csv('../data/dnafeat/*.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, 1:])
X_scaled32 = X_scaled.astype('float32').to_dask_array(lengths=True)

n_rows, n_cols = X_scaled32.shape
X_scaled32 = X_scaled32.rechunk((2000, n_cols))
# Removed unused variable `id_arr`.

da.to_hdf5(
    'dna_features.h5',
    {'/dnafeat': X_scaled32},      
    compression='lzf',
    chunks=(2000, n_cols)
)

id_list = df['id'].compute().tolist()
with h5py.File('dna_features.h5', 'a') as f:
    # define a variable‐length UTF-8 string dtype
    str_dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset(
        '/id',
        data=id_list,
        dtype=str_dt
    )