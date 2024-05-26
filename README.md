[![test](https://github.com/alanmeeson/lancedb-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/alanmeeson/lancedb-haystack/actions/workflows/test.yml)

# LanceDB Haystack Document store

LanceDB-Haystack is an embedded [LanceDB](https://lancedb.github.io/lancedb/) backed Document Store for 
[Haystack 2.X](https://github.com/deepset-ai/haystack/) intended for prototyping or small systems.

## Installation

The current simplest way to get LanceDB-Haystack is to install from GitHub via pip:

```pip install git+https://github.com/alanmeeson/lancedb-haystack.git```

## Usage

See the example in `examples/pipeline-usage.ipynb`

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
dist/lancedb_haystack-0.0.1.tar.gz

[wheel]
dist/lancedb_haystack-0.0.1-py3-none-any.whl
```

### Roadmap

In no particular order:
- **Write more documentation**
  
  There's docstrings in the code, but there's an outstanding task to configure pydoc or similar to produce some api
  docs.  Also, documentation to explain the data model and how it works would be a good idea.

- **Figure out if it's possible to have LanceDB work with dynamic metadata**

  Currently this implementation is limited to having only metadata which is defined in the metadata_schema.  It would be
  nice to be able to infer a schema from the first document to be added, or even better, be able to just have arbitrary
  metadata, rather than having to specify it all up front.

- **Expand the supported metadata types**
  
  As noted the metadata section requires a pyarrow schema;  not all of the types have been tested, and may not all be 
  supported.  It would be good to try out a few more to see if they're supported, and perhaps add those that aren't. 

## Limitations

The DocumentStore requires a pyarrow StructType to be specified as the schema for the metadata dict.  This should cover
all metadata fields which may appear in any of the documents you want to store.

Currently the system supports the basic datatypes (ints, floats, bools, strings, etc.)  as well as structs.  Others may
work, but haven't been tested.