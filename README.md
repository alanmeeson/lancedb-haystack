[![test](https://github.com/alanmeeson/lancedb-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/alanmeeson/lancedb-haystack/actions/workflows/test.yml)

# LanceDB Haystack Document store

LanceDB-Haystack is an embedded [LanceDB](https://lancedb.github.io/lancedb/) backed Document Store for 
[Haystack 2.X](https://github.com/deepset-ai/haystack/).

## Installation

The current simplest way to get LanceDB-Haystack is to install from GitHub via pip:

```pip install git+https://github.com/alanmeeson/lancedb-haystack.git```

## Usage

```python
import pyarrow as pa
from lancedb_haystack import LanceDBDocumentStore
from lancedb_haystack import LanceDBEmbeddingRetriever, LanceDBFTSRetriever

# Declare the metadata fields schema, this lets us filter using it.
# See: https://arrow.apache.org/docs/python/api/datatypes.html
metadata_schema = pa.struct([
  ('title', pa.string()),    
  ('publication_date', pa.timestamp('s')),
  ('page_number', pa.int32()),
  ('topics', pa.list_(pa.string()))
])

# Create the DocumentStore
document_store = LanceDBDocumentStore(
  database='my_database', 
  table_name="documents", 
  metadata_schema=metadata_schema, 
  embedding_dims=384
)

# Create an embedding retriever
embedding_retriever = LanceDBEmbeddingRetriever(document_store)

# Create a Full Text Search retriever
fts_retriever = LanceDBFTSRetriever(document_store)
```

See also `examples/pipeline-usage.ipynb` for a full worked example.

## Development

### Test

You can use `hatch` to run the linters:

```console
~$ hatch run lint:all
cmd [1] | ruff .
cmd [2] | black --check --diff .
All done! ✨ 🍰 ✨
6 files would be left unchanged.
cmd [3] | mypy --install-types --non-interactive src/lancedb_haystack tests
Success: no issues found in 6 source files
```

Similar for running the tests:

```console
~$ hatch run cov
cmd [1] | coverage run -m pytest tests
...
```

### Build

To build the package you can use `hatch`:

```console
~$ hatch build
[sdist]
dist/lancedb_haystack-0.1.0.tar.gz

[wheel]
dist/lancedb_haystack-0.1.0-py3-none-any.whl
```

### Document

To build the api docs run the following:

```console
~$ cd docs
~$ make clean
~$ make build
```

### Roadmap

In no particular order:

- **Figure out if it's possible to have LanceDB work with dynamic metadata**

  Currently, this implementation is limited to having only metadata which is defined in the metadata_schema.  It would be
  nice to be able to infer a schema from the first document to be added, or even better, be able to just have arbitrary
  metadata, rather than having to specify it all up front.

- **Expand the supported metadata types**
  
  As noted the metadata section requires a pyarrow schema;  not all of the types have been tested, and may not all be 
  supported.  It would be good to try out a few more to see if they're supported, and perhaps add those that aren't. 

## Limitations

The DocumentStore requires a pyarrow StructType to be specified as the schema for the metadata dict.  This should cover
all metadata fields which may appear in any of the documents you want to store.

Currently, the system supports the basic datatypes (ints, floats, bools, strings, etc.)  as well as structs and lists.  
Others may work, but haven't been tested.