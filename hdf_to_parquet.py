# Based on https://github.com/david4096/celldb/blob/master/etl/h5_to_tsv.py
#
# Start a Spark shell (`pyspark2 --master local[8]` on CDH) then run the following
#

# First idea was to use parallelize to read everything into memory, but this doesn't
# work - we run out of memory.

# The following still runs locally (to read the local file), but doesn't use so much
# memory. It took me 2.3 hours to run.
# https://stackoverflow.com/questions/31009951/loading-bigger-than-memory-hdf5-file-in-pyspark

'''
!pip install h5py
'''

import h5py
from pyspark.mllib.linalg import Vectors
from pyspark.sql.types import *

h5file_path="1M_neurons_filtered_gene_bc_matrices_h5.h5"

hF = None

def readchunk(k, shards):
    global hF
    if hF is None:
        hF = h5py.File(h5file_path)
    group = "mm10"
    indptr = hF[group +"/indptr"]
    indices = hF[group + "/indices"]
    data = hF[group + "/data"]
    genes = hF[group + "/genes"]
    gene_names = hF[group + "/gene_names"]
    barcodes = hF[group + "/barcodes"]
    shape = hF[group + "/shape"]
    rowN = shape[0]
    colN = shape[1]
    counter_indptr_size = rowN
    numFeatures = rowN
    if k == (shards - 1):
        to = len(barcodes)
    else:
        to = (k + 1) * len(barcodes) / (shards - 1)
    output = []
    for i in range (k * len(barcodes) / (shards - 1), to):
        barcode = barcodes[i]
        indices_range = indices[indptr[i]:indptr[i+1]]
        data_range = data[indptr[i]:indptr[i+1]]
        pairs = zip(indices_range, data_range) # zip to ensure that indices are in ascending order
        output.append((barcode.replace('-',''), Vectors.sparse(numFeatures, pairs)))
    return output

total_chunks = 320
foo = sc.parallelize(range(0,total_chunks), total_chunks).flatMap(lambda k: readchunk(k, total_chunks))

rowRDD = foo.map(lambda (id, vec): (id, vec.indices.tolist(), vec.values.tolist()))

schema = StructType([StructField("id", StringType(), False),
                     StructField("idx", ArrayType(IntegerType(), False), False),
                     StructField("quant", ArrayType(DoubleType(), False), False)])
df = spark.createDataFrame(rowRDD, schema)
df.write.parquet("celldb1m")

if hF is None:
  hF = h5py.File(h5file_path)
  
group = "mm10"
genes = hF[group + "/genes"]
gene_names = hF[group + "/gene_names"]
gene_names[27228] # 'Malat1'