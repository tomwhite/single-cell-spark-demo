# https://stackoverflow.com/questions/46157709/converting-hdf5-to-parquet-without-loading-into-memory

# This just runs as a local python program. However, pandas doesn't seem able to read the
# datasets from the h5 files.

# pip install pandas pyarrow tables

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_hdf5_to_parquet(h5_file, parquet_file, chunksize=100000):
    stream = pd.read_hdf(h5_file, chunksize=chunksize)
    for i, chunk in enumerate(stream):
        print("Chunk {}".format(i))
        if i == 0:
            # Infer schema and open parquet file on first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')
        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)
    parquet_writer.close()

convert_hdf5_to_parquet('test1.h5',
                        'test1.parquet')

convert_hdf5_to_parquet('1M_neurons_filtered_gene_bc_matrices_h5.h5',
                        '1M_neurons_filtered_gene_bc_matrices_h5.parquet')

# File "/home/tom/hca/hca-virt/lib/python2.7/site-packages/pandas/io/pytables.py",  line 360, in read_hdf
# raise ValueError('No dataset in HDF5 file.')
# ValueError: No dataset in HDF5 file.
