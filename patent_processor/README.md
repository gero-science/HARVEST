# Patent Processor Module

Module for extracting and structuring data from patent ZIP archives, with a focus on chemical compounds.

## Key features

- **Reliable data structures**: Uses `Pydantic` to define clear, typed structures (`PatentDocument`, `ChemistryNode`).
- **RDKit integration**: Processes `.MOL` files and converts them to **SMILES** and **InChIKey** for standardization and chemical data enrichment.
- **CDX structures**: Converts `.CDX` files from the same archive into a separate structure source (`PatentDocument.cdx_data`), exported downstream as `smiles_cdx` / `inchi_key_cdx`.
- **Async and streaming processing**:
    - **Async API** (`async_iter_documents`) for integration into high-performance event-loop pipelines.
    - **Lazy (streaming) processing** of large ZIP archives using `zipfile`, minimizing memory usage.
- **Safe XML parsing**: Robust parsing of patent XML content using `lxml`.
- **Advanced data extraction**:
    - **Chemical compounds**: Extraction of chemical compound data from `<chemistry>` tags and enrichment with **SMILES strings** from `.MOL` files. This links text mentions to specific molecular structures.
- **Context extraction**: Finds and extracts text paragraphs surrounding chemical structures.
- **Flexible JSON export**: Convenient saving of extracted chemistry nodes to JSON files via `NodeExporter`.

## Module architecture

The module consists of several key components following separation of concerns:

- `patent_processor.py`: Contains the `PatentProcessor` class — the main facade and entry point. It provides a simple API for sync and async document iteration, encapsulating extractor work internally.
- `zip_source.py`: `ZipPatentSource` reads data from ZIP archives, including **XML patents and associated `.MOL` and `.CDX` files**. It uses `RDKit` for chemical structure processing. CDX parsing can be disabled with `parse_cdx=False`.
- `cdx_extractor.py`: shared CDX helpers (`parse_cdx_bytes`, `extract_chem_num_from_cdx_filename`), used by `zip_source.py`. The actual CDX decoding lives in the external [`cdx_reader`](https://github.com/gero-science/cdx_reader) package.
- `xml_patent_parser.py`: Simple utility parser that safely converts an XML string into an `lxml` object.
- `chemistry_extractor.py`: `ChemistryExtractor` for extracting and structuring chemical compound data and context.
- `node_exporter.py`: `NodeExporter` saves extracted `ChemistryNode` data to JSON files (mainly used for debugging).
- `data_structures.py`: Defines core data structures (`PatentDocument`, `ChemistryNode`) using `Pydantic`.

## Usage

### Async usage (recommended)

This approach is ideal for integration into the project's main async pipeline. `PatentProcessor` handles data extraction internally.

```python
import asyncio
import logging
from patent_processor import PatentProcessor, NodeExporter

# (Configure logging)
# logging.basicConfig(level=logging.INFO)

async def process_patents_async():
    # 1. Initialize processor
    zip_file = "path/to/patents.zip"
    processor = PatentProcessor(zip_file, batch_size=100)
    
    # (Optional) Initialize exporter for saving results
    exporter = NodeExporter()
    output_dir = "output_data"

    # 2. Process all patents asynchronously
    total_chemistry_found = 0
    try:
        # processor.async_iter_documents() already returns enriched documents
        async for document in processor.async_iter_documents():
            if document.parsing_errors:
                logging.warning(f"Skipping patent {document.patent_id} due to parsing errors.")
                continue

            if document.chemistry_nodes:
                logging.info(f"Found {len(document.chemistry_nodes)} chemistry nodes in patent {document.patent_id}")
                total_chemistry_found += len(document.chemistry_nodes)
                
                # Example: save chemistry nodes to JSON for debugging
                await exporter.async_export_chemistry_as_single_file(
                    document.chemistry_nodes,
                    output_dir=f"{output_dir}/{document.patent_id}/chemistry",
                )

        print(f"Processing complete. Total chemistry nodes found: {total_chemistry_found}")

    except FileNotFoundError:
        print(f"Error: File not found at path {zip_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run async function
# asyncio.run(process_patents_async())
```

### Sync usage

Suitable for simple scripts, data analysis, or debugging.

```python
import logging
from patent_processor import PatentProcessor, NodeExporter

# 1. Initialize components
processor = PatentProcessor("path/to/patents.zip")
exporter = NodeExporter()
output_dir = "output_data"

# 2. Sync processing
for document in processor.iter_documents():
    if document.chemistry_nodes:
        # Sync save
        exporter.export_chemistry_as_single_file(
            document.chemistry_nodes,
            output_dir=f"{output_dir}/{document.patent_id}/chemistry",
        )
```

## Dependencies

All required dependencies should be installed from `requirements.txt` at the project root.

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Key dependencies for this module:
- `lxml`: high-performance XML parsing.
- `pydantic`: data validation.
- `rdkit-pypi`: chemical structure processing and SMILES generation.
