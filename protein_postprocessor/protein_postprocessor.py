import asyncio
import json
import logging
import sys
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.call_llm import LLM
from .config_postprocess import DEFAULT_FASTA_PATH, ConfigProteinLLM


class ProteinPostprocessor:
    def __init__(self, debug_mode: bool = False, debug_output_dir: str = None,
                 zip_test_dir: str = None, fasta_path=None):
        """
        Args:
            fasta_path: UniProt FASTA to resolve sequences against. Defaults to
                DEFAULT_FASTA_PATH; `pipeline.py --protein-data-path` overrides it.
        """
        self.config = ConfigProteinLLM()
        self.fasta_path = Path(fasta_path) if fasta_path else DEFAULT_FASTA_PATH
        self.llm = LLM.from_config(self.config, logging.getLogger(__name__), debug_mode, debug_output_dir)
        self.resolver = None  # Will be initialized asynchronously
        self.zip_test_dir = zip_test_dir

    async def close(self):
        """Close the LLM aiohttp session."""
        await self.llm.close()

    async def _ensure_resolver_initialized(self):
        """Initialize FastaGeneResolver if not already initialized."""
        if self.resolver is None:
            from protein_resolver.fasta_gene_resolver import FastaGeneResolver
            self.resolver = await FastaGeneResolver.create(self.fasta_path)
    
    
    def _load_stage1_context(self, patent_id: str) -> str:
        """
        Load Stage 1 TSV as context for the LLM.
        
        Args:
            patent_id: Patent ID
        
        Returns:
            Context string from Stage 1 (protein_target_name + assay_description)
        """
        if not self.zip_test_dir:
            return ""
        
        stage1_path = Path(self.zip_test_dir) / patent_id / f"{patent_id}_agent1_stage1_targets.tsv"
        if not stage1_path.exists():
            logging.debug(f"Stage 1 file not found: {stage1_path}")
            return ""
        
        try:
            lines = stage1_path.read_text(encoding='utf-8').splitlines()
            if len(lines) < 2:
                return ""
            
            # Build compact context
            context_lines = []
            seen_proteins = set()  # Deduplication
            
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 8:
                    protein = parts[1]  # protein_target_name
                    
                    # Deduplicate by protein name
                    if protein in seen_proteins:
                        continue
                    seen_proteins.add(protein)
                    
                    description = parts[4][:200]  # assay_description (truncated)
                    assay_type = parts[6]  # assay (assay type)
                    organism = parts[7]  # organism
                    
                    context_lines.append(f"- {protein} ({organism}): {assay_type}. {description}...")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logging.warning(f"Error reading Stage 1 for {patent_id}: {e}")
            return ""
    
    async def postprocess(
        self, 
        protein_data: list[dict[str, str]],
        stage1_context: str = ""
    ) -> list[dict]:
        """
        Postprocess protein names to get gene symbol, species, sequence, accession, and uniprot_id.
        
        Args:
            protein_data: List of dicts with 'protein_name' and 'organism' keys
                Example: [
                    {"protein_name": "11β-HSD1", "organism": "Human"},
                    {"protein_name": "5-HT7 receptor", "organism": "Rat"}
                ]
            stage1_context: Context from Stage 1 (assay descriptions) to assist the LLM
        
        Returns:
            List of protein result objects:
            [
                {
                    "protein_name": "EGFR",
                    "organism": "human",
                    "organism_scientific": "Homo sapiens",
                    "gene": "EGFR",
                    "sequence": "MKTAY...",
                    "accession": "P00533",
                    "uniprot_id": "EGFR_HUMAN"
                },
                ...
            ]
        """
        # Ensure resolver is initialized
        await self._ensure_resolver_initialized()
        
        # Build LLM input: "EGFR (human); TNF (mouse); FabI (s. aureus)"
        input_text = "; ".join(
            f"{p['protein_name'].replace(chr(92), '')} ({p['organism']})" 
            for p in protein_data
        )
        
        # Get LLM response
        if stage1_context:
            user_prompt = self.config.USER_PROMPT_WITH_CONTEXT.format(
                stage1_context=stage1_context,
                text_chunk=input_text
            )
        else:
            user_prompt = self.config.USER_PROMPT_SIMPLE.format(text_chunk=input_text)
        
        raw, _ = await self.llm.async_call_llm(user_prompt, self.config.SYSTEM_PROMPT)
        llm_results = self._parse_llm_response(raw)
        
        # Build result list
        result = []
        
        for item in llm_results:
            protein_name = item.get("protein")
            organism = item.get("organism", "")
            gene = item.get("gene")
            species = item.get("species") or organism  # Fallback to organism if species=None
            
            if not protein_name:
                continue
            
            # Base result object
            protein_result = {
                "protein_name": protein_name,
                "organism": organism,
                "organism_scientific": species,
                "gene": gene,
                "sequence": None,
                "accession": None,
                "uniprot_id": None
            }
            
            if gene is None:
                result.append(protein_result)
                continue
            
            # Resolve through FastaGeneResolver using gene + species (scientific name)
            seq_record = self.resolver.resolve(gene, species)
            
            if seq_record:
                protein_result["sequence"] = str(seq_record.seq)
                protein_result["accession"] = self.resolver.extract_accession_from_id(seq_record.id)
                protein_result["uniprot_id"] = self.resolver.extract_uniprot_from_id(seq_record.id)
            else:
                logging.debug(f"No sequence found for {gene} ({species})")
            
            result.append(protein_result)
        
        return result
    
    async def postprocess_complex(
        self, 
        complex_data: list[dict[str, str]],
        stage1_context: str = ""
    ) -> list[dict]:
        """
        Process protein complexes by extracting all genes for each complex.
        
        Args:
            complex_data: List of dicts with 'complex_name' and 'organism' keys
                Example: [
                    {"complex_name": "IL-12 receptor signaling", "organism": "human"},
                    {"complex_name": "NF-kB pathway", "organism": "mouse"}
                ]
            stage1_context: Stage 1 context to assist the LLM
        
        Returns:
            List of complex result objects with nested proteins:
            [
                {
                    "complex_name": "IL-12 receptor signaling",
                    "organism": "human",
                    "species": "Homo sapiens",
                    "proteins": [
                        {"gene": "TYK2", "sequence": "...", "accession": "...", "uniprot_id": "..."},
                        ...
                    ]
                }
            ]
        """
        await self._ensure_resolver_initialized()
        
        # Build LLM input: "IL-12 receptor signaling (human); NF-kB pathway (mouse)"
        input_text = "; ".join(
            f"{c['complex_name'].replace(chr(92), '')} ({c['organism']})" 
            for c in complex_data
        )
        
        # Get LLM response
        if stage1_context:
            user_prompt = self.config.USER_PROMPT_COMPLEX_WITH_CONTEXT.format(
                stage1_context=stage1_context,
                text_chunk=input_text
            )
        else:
            user_prompt = self.config.USER_PROMPT_COMPLEX_SIMPLE.format(text_chunk=input_text)
        
        raw, _ = await self.llm.async_call_llm(user_prompt, self.config.SYSTEM_PROMPT_COMPLEX)
        llm_results = self._parse_llm_response(raw)
        
        # Build result list with resolved proteins
        result = []
        
        for item in llm_results:
            complex_name = item.get("complex_name")
            organism = item.get("organism", "")
            species = item.get("species") or organism  # species at complex level
            genes = item.get("genes", [])  # genes as a string array
            
            if not complex_name:
                continue
            
            # Resolve each gene in the complex
            resolved_proteins = []
            for gene in genes:
                if not gene:
                    continue
                
                protein_result = {
                    "gene": gene,
                    "sequence": None,
                    "accession": None,
                    "uniprot_id": None
                }
                
                seq_record = self.resolver.resolve(gene, species)
                if seq_record:
                    protein_result["sequence"] = str(seq_record.seq)
                    protein_result["accession"] = self.resolver.extract_accession_from_id(seq_record.id)
                    protein_result["uniprot_id"] = self.resolver.extract_uniprot_from_id(seq_record.id)
                
                resolved_proteins.append(protein_result)
            
            result.append({
                "complex_name": complex_name,
                "organism": organism,
                "species": species,
                "proteins": resolved_proteins
            })
        
        return result
    
    def _parse_llm_response(self, response: str) -> list[dict]:
        """
        Parse the LLM response in the new format.
        
        Expected format:
        [
            {"protein": "EGFR", "organism": "human", "gene": "EGFR", "species": "Homo sapiens"},
            ...
        ]
        """
        try:
            data = json.loads(response)
            
            if not isinstance(data, list):
                logging.error(f"Expected list, got {type(data).__name__}")
                return []
            
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return []
    
    # ===== Per-patent processing =====
    
    async def _process_single_proteins(
        self, 
        patent_id: str, 
        input_file: Path,
        use_stage1_context: bool = False,
        reprocess_all: bool = False
    ) -> dict:
        """
        Process a patent's individual proteins (is_complex != "Y").
        
        Args:
            patent_id: Patent ID
            input_file: Path to *_resolved.json
            use_stage1_context: Whether to load context from Stage 1
            reprocess_all: Process all proteins, ignoring existing sequence
        
        Returns:
            Dict with patent_id and a proteins array
        """
        data = json.loads(input_file.read_text(encoding='utf-8'))
        
        if not data:
            return {"patent_id": patent_id, "proteins": []}
        
        # Extract unique individual proteins (NOT complexes)
        seen = set()
        proteins = []
        for entry in data:
            # Skip complexes
            if entry.get("is_complex") == "Y":
                continue
            
            # Skip already resolved entries (unless reprocess_all)
            if not reprocess_all and entry.get("protein_sequence"):
                continue
            
            protein_name = entry.get("protein_target_name")
            organism = entry.get("organism") or "human"
            if protein_name and (protein_name, organism) not in seen:
                seen.add((protein_name, organism))
                proteins.append({
                    "protein_name": protein_name,
                    "organism": organism,
                    "patent_id": patent_id
                })
        
        if not proteins:
            return {"patent_id": patent_id, "proteins": []}
        
        # Load Stage 1 context only when the flag is enabled
        stage1_context = ""
        if use_stage1_context:
            stage1_context = self._load_stage1_context(patent_id)
        
        result = await self.postprocess(proteins, stage1_context)
        return {"patent_id": patent_id, "proteins": result}
    
    async def _process_complexes(
        self, 
        patent_id: str, 
        input_file: Path,
        use_stage1_context: bool = False,
        reprocess_all: bool = False
    ) -> dict:
        """
        Process a patent's complexes (is_complex == "Y").
        
        Args:
            patent_id: Patent ID
            input_file: Path to *_resolved.json
            use_stage1_context: Whether to load context from Stage 1
            reprocess_all: Process all complexes, ignoring existing sequence
        
        Returns:
            Dict with patent_id and a complexes array
        """
        data = json.loads(input_file.read_text(encoding='utf-8'))
        
        if not data:
            return {"patent_id": patent_id, "complexes": []}
        
        # Extract unique complexes
        seen = set()
        complexes = []
        for entry in data:
            # Complexes only
            if entry.get("is_complex") != "Y":
                continue
            
            # Skip already resolved entries (unless reprocess_all)
            if not reprocess_all and entry.get("protein_sequence"):
                continue
            
            complex_name = entry.get("protein_target_name")
            organism = entry.get("organism") or "human"
            if complex_name and (complex_name, organism) not in seen:
                seen.add((complex_name, organism))
                complexes.append({
                    "complex_name": complex_name,
                    "organism": organism,
                    "patent_id": patent_id
                })
        
        if not complexes:
            return {"patent_id": patent_id, "complexes": []}
        
        # Load Stage 1 context only when the flag is enabled
        stage1_context = ""
        if use_stage1_context:
            stage1_context = self._load_stage1_context(patent_id)
        
        result = await self.postprocess_complex(complexes, stage1_context)
        return {"patent_id": patent_id, "complexes": result}
    
    async def _worker(
        self, 
        queue: asyncio.Queue, 
        counter_lock: asyncio.Lock,
        counter: dict,
        use_stage1_context: bool,
        reprocess_all: bool,
        process_singles: bool,
        process_complexes: bool,
        worker_id: int
    ):
        """
        Worker for parallel patent processing.
        
        Args:
            queue: Task queue
            counter_lock: Lock for counter synchronization
            counter: Counter dict {"done": N, "total": M}
            use_stage1_context: Pass Stage 1 context to the LLM
            reprocess_all: Process all proteins
            process_singles: Process individual proteins
            process_complexes: Process complexes
            worker_id: Worker ID for logging
        """
        while True:
            task = await queue.get()
            
            if task is None:  # Shutdown signal
                queue.task_done()
                break
            
            patent_id, input_file, needs_singles, needs_complexes = task
            
            try:
                singles_resolved = 0
                singles_total = 0
                complexes_resolved = 0
                complexes_total = 0
                
                # Process individual proteins
                if process_singles and needs_singles:
                    result_singles = await self._process_single_proteins(
                        patent_id, input_file, use_stage1_context, reprocess_all
                    )
                    output_file = input_file.parent / "single_proteins.json"
                    output_file.write_text(
                        json.dumps(result_singles, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )
                    proteins_list = result_singles.get("proteins", [])
                    singles_resolved = sum(1 for p in proteins_list if p.get("sequence"))
                    singles_total = len(proteins_list)
                
                # Process complexes
                if process_complexes and needs_complexes:
                    result_complexes = await self._process_complexes(
                        patent_id, input_file, use_stage1_context, reprocess_all
                    )
                    output_file = input_file.parent / "protein_complexes.json"
                    output_file.write_text(
                        json.dumps(result_complexes, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )
                    # Count proteins inside complexes
                    for cpx in result_complexes.get("complexes", []):
                        prots = cpx.get("proteins", [])
                        complexes_total += len(prots)
                        complexes_resolved += sum(1 for p in prots if p.get("sequence"))
                
                # Update counter
                async with counter_lock:
                    counter["done"] += 1
                    done = counter["done"]
                    total = counter["total"]
                
                # Log result
                parts = []
                if process_singles and needs_singles:
                    parts.append(f"proteins {singles_resolved}/{singles_total}")
                if process_complexes and needs_complexes:
                    parts.append(f"complexes {complexes_resolved}/{complexes_total}")
                
                logging.info(f"[{done}/{total}] {patent_id}: {', '.join(parts) or 'skipped'}")
                
            except Exception as e:
                logging.error(f"[Worker {worker_id}] Error processing {patent_id}: {e}")
                async with counter_lock:
                    counter["done"] += 1
            finally:
                queue.task_done()
    
    async def process_results_dir(
        self, 
        results_dir: str,
        num_workers: int = 5,
        use_stage1_context: bool = False,
        reprocess_all: bool = False,
        force: bool = False,
        singles_only: bool = False,
        complexes_only: bool = False
    ):
        """
        Process a pipeline results directory in parallel.
        
        Reads *_resolved.json from each patent folder and extracts proteins.
        Results are saved to:
        - {patent_dir}/single_proteins.json (individual proteins)
        - {patent_dir}/protein_complexes.json (complexes)
        
        Args:
            results_dir: Pipeline results directory
            num_workers: Number of parallel workers (default: 5)
            use_stage1_context: Pass Stage 1 context to the LLM (default: False)
            reprocess_all: Process all proteins, ignoring existing sequence
            force: Reprocess even if output files already exist
            singles_only: Process individual proteins only
            complexes_only: Process complexes only
        """
        results_path = Path(results_dir)
        
        # Determine what to process
        process_singles = not complexes_only  # Default: yes
        process_complexes = not singles_only  # Default: yes
        
        # Update zip_test_dir for stage1_context
        if not self.zip_test_dir:
            self.zip_test_dir = str(results_path)
        
        # Collect patents to process
        patents_to_process = []
        skipped_singles = 0
        skipped_complexes = 0
        
        for patent_dir in sorted(results_path.iterdir()):
            if not patent_dir.is_dir():
                continue
            
            patent_id = patent_dir.name
            
            # Always read *_resolved.json
            input_file = patent_dir / f"{patent_id}_resolved.json"
            if not input_file.exists():
                continue
            
            # Check whether singles need processing
            singles_file = patent_dir / "single_proteins.json"
            needs_singles = process_singles and (force or not singles_file.exists())
            if process_singles and singles_file.exists() and not force:
                skipped_singles += 1
            
            # Check whether complexes need processing
            complexes_file = patent_dir / "protein_complexes.json"
            needs_complexes = process_complexes and (force or not complexes_file.exists())
            if process_complexes and complexes_file.exists() and not force:
                skipped_complexes += 1
            
            # Enqueue if there is work to do
            if needs_singles or needs_complexes:
                patents_to_process.append((patent_id, input_file, needs_singles, needs_complexes))
        
        if skipped_singles > 0:
            logging.info(f"Skipped proteins (already processed): {skipped_singles}")
        if skipped_complexes > 0:
            logging.info(f"Skipped complexes (already processed): {skipped_complexes}")
        
        if not patents_to_process:
            logging.info("All patents are already processed or there are no files to process.")
            return
        
        mode_parts = []
        if process_singles:
            mode_parts.append("proteins")
        if process_complexes:
            mode_parts.append("complexes")
        
        logging.info(
            f"Patents to process: {len(patents_to_process)}, "
            f"workers: {num_workers}, "
            f"mode: {' + '.join(mode_parts)}, "
            f"Stage 1 context: {use_stage1_context}"
        )
        
        # Initialize resolver upfront (once for all workers)
        await self._ensure_resolver_initialized()
        
        # Create queue and synchronization primitives
        queue = asyncio.Queue()
        counter_lock = asyncio.Lock()
        counter = {"done": 0, "total": len(patents_to_process)}
        
        # Start workers
        workers = [
            asyncio.create_task(
                self._worker(
                    queue, counter_lock, counter,
                    use_stage1_context, reprocess_all,
                    process_singles, process_complexes,
                    worker_id=i
                )
            )
            for i in range(num_workers)
        ]
        
        # Enqueue tasks
        for task in patents_to_process:
            await queue.put(task)
        
        # Shutdown signal for each worker
        for _ in range(num_workers):
            await queue.put(None)
        
        # Wait for all tasks to finish
        await queue.join()
        
        # Cancel workers (in case any remain)
        for w in workers:
            w.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        logging.info(f"Processing complete: {counter['done']}/{counter['total']} patents")
