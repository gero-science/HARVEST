"""
Main processor for patent data from ZIP archives.
Provides a simple API for lazy patent processing.
"""

import asyncio
import logging
from typing import Iterator, Optional, AsyncIterator, Union
from .data_structures import PatentDocument
from .zip_source import ZipPatentSource
from .chemistry_extractor import ChemistryExtractor


class PatentProcessor:
    """
    Main processor for patent data from ZIP archives.
    
    Provides lazy (streaming) data processing, allowing work with archives
    of any size with minimal memory consumption.
    
    Usage examples:
        # Simple iteration over documents
        processor = PatentProcessor('patents.zip')
        for doc in processor.iter_documents():
            print(f"Patent: {doc.patent_id}, Title: {doc.title}")
        
        # Get document count
        count = processor.get_document_count()
        print(f"Total documents: {count}")
    """
    
    def __init__(
        self,
        zip_path: str,
        batch_size: int = 1000,
        parse_cdx: bool = True,
    ):
        """
        Initialize the processor.
        
        Args:
            zip_path: Path to the ZIP file with patents
            batch_size: Batch size for reading (memory optimization)
            parse_cdx: Parse CDX files from the archive alongside MOL files
        """
        self.zip_path = zip_path
        self.batch_size = batch_size
        self.parse_cdx = parse_cdx
        self._source = None
        self.logger = logging.getLogger(__name__)
        
        self.chemistry_extractor = ChemistryExtractor()
        
        self.logger.info(f"Initialized PatentProcessor for: {zip_path}")
    
    def _enrich_chemistry_nodes(self, document: PatentDocument) -> None:
        """
        Enrich ChemistryNode with SMILES data from ZipPatentSource cache (Iteration 2).
        Also adds is_scaffold marker for incomplete molecules (scaffolds).
        
        Args:
            document: Patent document to enrich
        """
        if not document.chemistry_nodes:
            return
        
        # Get SMILES and scaffold cache for this patent
        mol_data = self.source.get_smiles_for_patent(document.patent_id)
        
        if not mol_data:
            self.logger.debug(f"No MOL data cache found for patent {document.patent_id}")
            return
        
        enriched_count = 0
        scaffold_count = 0
        for node in document.chemistry_nodes:
            if not node.chem_num:
                self.logger.debug(f"No chem_num found for chemistry node {node.chemistry_id}")
                continue
            
            # Get data for this chem_num
            data = mol_data.get(node.chem_num)
            if not data:
                continue
            
            # Check whether this is a scaffold (incomplete molecule)
            if data.get('is_scaffold'):
                node.is_scaffold = True
                node.mol_filename = data.get('filename')
                scaffold_count += 1
                self.logger.debug(f"Marked chemistry node {node.chemistry_id} (chem_num: {node.chem_num}) as scaffold (incomplete molecule)")
                continue  # Skip SMILES enrichment for scaffolds
            
            # Enrich with SMILES data (full molecules only)
            if data.get('smiles'):
                node.smiles = data.get('smiles')
                node.mol_filename = data.get('filename')
                node.inchikey = data.get('inchikey')  # Add InChI key
                node.molecular_weight = data.get('mol_weight')
                node.is_encoding_error = data.get('is_encoding_error')
                enriched_count += 1
                self.logger.debug(f"Enriched chemistry node {node.chemistry_id} (chem_num: {node.chem_num}) with SMILES: {node.smiles}, InChIKey: {node.inchikey}, mol weight: {node.molecular_weight}")
        
        if enriched_count > 0 or scaffold_count > 0:
            self.logger.info(f"Enriched {enriched_count}/{len(document.chemistry_nodes)} chemistry nodes with SMILES, "
                           f"marked {scaffold_count} as scaffolds for patent {document.patent_id}")
    
    def _attach_cdx_data(self, document: PatentDocument) -> None:
        """
        Attach CDX structures from the ZipPatentSource cache to the document.
        
        CDX stays a separate structure source and does not overwrite
        ChemistryNode.smiles from MOL files.
        
        Args:
            document: Patent document to populate
        """
        cdx_data = self.source.get_cdx_for_patent(document.patent_id)
        if not cdx_data:
            return
        
        document.cdx_data = cdx_data
        self.logger.debug(f"Attached {len(cdx_data)} CDX entries for patent {document.patent_id}")
    
    
    @property
    def source(self) -> ZipPatentSource:
        """Lazy initialization of the data source"""
        if self._source is None:
            self.logger.debug("Initializing ZipPatentSource")
            self._source = ZipPatentSource(self.zip_path, self.batch_size, parse_cdx=self.parse_cdx)
        return self._source
    
    def iter_documents(self) -> Iterator[PatentDocument]:
        """
        Iterate over all patent documents, extracting chemistry.
        
        Provides lazy processing - data is read and processed on demand,
        without loading the entire file into memory.
        
        Yields:
            PatentDocument: Fully populated document with metadata, XML, and chemistry
        """
        for document in self.source.iter_patents():
            if document.xml_root is not None:
                document.chemistry_nodes = self.chemistry_extractor.extract_all_nodes(document)
                
                # NEW STEP: Enrich ChemistryNode with SMILES data (Iteration 2)
                self._enrich_chemistry_nodes(document)
                self._attach_cdx_data(document)
            
            yield document
    
    async def async_iter_documents(self) -> AsyncIterator[PatentDocument]:
        """
        ASYNCHRONOUS version of the iterator over all patent documents with chemistry extraction.
        
        Offloads blocking read, parse, and extraction operations to a separate thread,
        allowing the event loop to continue during I/O operations.
        
        Yields:
            PatentDocument: Fully populated document with metadata, XML, and chemistry
        """
        self.logger.debug("Starting async document iteration with data enrichment")
        
        # Get synchronous iterator in a separate thread
        def get_sync_iterator():
            return self.source.iter_patents()
        
        # Create iterator in thread pool
        sync_iterator = await asyncio.to_thread(get_sync_iterator)
        
        # Iterate over documents asynchronously
        while True:
            try:
                # Get next document in a separate thread
                doc = await asyncio.to_thread(next, sync_iterator, None)
                if doc is None:
                    break
                
                # Asynchronous data enrichment
                if doc.xml_root is not None:
                    doc.chemistry_nodes = await self.chemistry_extractor.async_extract_all_nodes(doc)
                    
                    # NEW STEP: Enrich ChemistryNode with SMILES data (Iteration 2)
                    await asyncio.to_thread(self._enrich_chemistry_nodes, doc)
                    await asyncio.to_thread(self._attach_cdx_data, doc)
                
                yield doc
            except StopIteration:
                break
            except Exception as e:
                self.logger.error(f"Error during async document iteration: {e}")
                break
        
        self.logger.debug("Completed async document iteration")
    
    def get_document_count(self) -> Optional[int]:
        """
        Return the total number of documents in the file.
        
        Returns:
            int: Number of documents or None on error
        """
        self.logger.debug("Getting document count")
        count = self.source.get_total_count()
        if count is not None:
            self.logger.info(f"Found {count} documents in file")
        else:
            self.logger.warning("Could not determine document count")
        return count
