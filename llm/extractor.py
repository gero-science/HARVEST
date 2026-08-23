"""
Simplified Agent 1 for bioactivity extraction from patents.

Passes the full patent XML to the LLM without preprocessing.
Designed for Gemini 2.5 (1M token context).
"""

import json
import logging
import re
from typing import Dict, List, Optional

from .continuation import ImprovedContinuationLogic
from patent_processor import PatentDocument
from .call_llm import LLM
from .config import ConfigLLM
from .prompts import load_prompt_file
from .stage_formats import StageFormats
from bioactivity_extraction.compound_alias import (
    extract_compound_number,
    normalize_compound_alias,
)
from bioactivity_extraction.artifacts import save_stage_responses
from bioactivity_extraction.continuation import request_continuation as request_continuation_basic
from bioactivity_extraction.stage1_filter import filter_stage1_for_stage2
from bioactivity_extraction.stage3_cleanup import (
    check_duplicate_chemical_ids,
    check_duplicate_iupac_names,
    remove_duplicate_chemical_ids,
    remove_duplicate_iupac_names,
)
from bioactivity_extraction.stage_merge import merge_with_assay_id
from bioactivity_extraction.structure_enrichment import enrich_with_smiles
from bioactivity_extraction.tsv import ensure_header_in_tsv, merge_tsv_parts
from bioactivity_extraction.usage import prepare_usage_stats_list
from bioactivity_extraction.validation import validate_item
from bioactivity_extraction.xml_sections import extract_relevant_xml_sections

try:
    from py2opsin import py2opsin
    PY2OPSIN_AVAILABLE = True
except ImportError:
    PY2OPSIN_AVAILABLE = False
    logging.warning("py2opsin not available. Install with 'pip install py2opsin' for IUPAC name resolution.")

improved_continuation = True

class BioactivityExtractor:
    """Simplified Agent 1: passes full XML to the LLM without preprocessing.

    Uses a three-stage approach with prompt caching:
    - Stage 1: Extract protein targets and assay descriptions (Cache Write)
    - Stage 2: Extract bioactivities with compound_alias (Cache Read ~10x cheaper)
    - Stage 3: Extract final compounds (Cache Read ~10x cheaper)

    OPTIMIZATION: Stage 3 uses a SEPARATE context (XML + Stage1) without Stage2 responses.
    This reduces input tokens by 10-20K and improves LLM output quality.
    The compound_alias list from Stage 2 is passed explicitly in the prompt.
    """
    
    # Constants for handling truncated responses
    MAX_CONTINUATIONS = 1
    
    # Flag controlling removal of duplicate IUPAC names
    REMOVE_DUPLICATE_IUPAC = False
    
    # Flag controlling removal of duplicate chemical_id values
    REMOVE_DUPLICATE_CHEMICAL_ID = True
    
    # Load prompts from files
    CONTINUATION_PROMPT = load_prompt_file('continuation_prompt.txt')
    SYSTEM_PROMPT = load_prompt_file('system_prompt.txt')
    USER_PROMPT_STAGE1 = load_prompt_file('user_prompt_stage1.txt')
    USER_PROMPT_STAGE2 = load_prompt_file('user_prompt_stage2.txt')
    USER_PROMPT_STAGE3 = load_prompt_file('user_prompt_stage3.txt')

    def __init__(self, config: ConfigLLM, llm: LLM):
        """
        Initialize the extractor.
        
        Args:
            config: LLM configuration
            llm: LLM instance for API calls
        """
        self.config = config
        self.llm = llm
        self.logger = logging.getLogger(__name__)

        # Initialize improved continuation logic when enabled
        if improved_continuation:
            improved_logic = ImprovedContinuationLogic(self.logger)
            # Wrapper method for improved logic
            async def _continuation_wrapper(messages, patent_id, stage_name, expected_columns):
                return await improved_logic.request_continuation_safe(
                    self.llm, messages, patent_id, stage_name, expected_columns
                )
            self.request_continuation = _continuation_wrapper
            self.logger.debug("Improved continuation logic enabled")
        else:
            # Wrapper method for standard logic
            async def _continuation_wrapper(messages, patent_id, stage_name, expected_columns):
                responses, usage_stats = await self._request_continuation(
                    messages=messages, patent_id=patent_id, stage_name=stage_name
                )
                # Return empty diagnostics for compatibility
                return responses, usage_stats, {}
            self.request_continuation = _continuation_wrapper
            self.logger.debug("Using standard continuation logic")

    def _extract_relevant_xml_sections(self, xml_root) -> str:
        return extract_relevant_xml_sections(xml_root, logger=self.logger)
    
    def _check_duplicate_iupac_names(self, compounds_data: List[Dict]) -> set:
        return check_duplicate_iupac_names(compounds_data, logger=self.logger)
    
    def _remove_duplicate_iupac_names(self, compounds_data: List[Dict], duplicate_names: set) -> List[Dict]:
        return remove_duplicate_iupac_names(compounds_data, duplicate_names)
    
    def _check_duplicate_chemical_ids(self, compounds_data: List[Dict]) -> set:
        return check_duplicate_chemical_ids(compounds_data, logger=self.logger)
    
    def _remove_duplicate_chemical_ids(self, compounds_data: List[Dict], duplicate_chem_ids: set) -> List[Dict]:
        return remove_duplicate_chemical_ids(compounds_data, duplicate_chem_ids)
    
    def _merge_tsv_parts(self, tsv_parts: List[str], stage_name: str, patent_id: str = "") -> List[Dict]:
        return merge_tsv_parts(tsv_parts, stage_name, patent_id, logger=self.logger)
    
    def _ensure_header_in_tsv(self, tsv_responses: List[str], expected_header: str, stage_name: str, patent_id: str = "") -> List[str]:
        return ensure_header_in_tsv(tsv_responses, expected_header, stage_name, patent_id, logger=self.logger)

    async def _request_continuation(
        self, 
        messages: List[Dict], 
        patent_id: str,
        stage_name: str,
        max_continuations: int = None
    ) -> tuple[List[str], List[Dict]]:
        """
        Request continuation if the response was truncated.
        
        Args:
            messages: Message list for the LLM (modified in-place during continuation)
            patent_id: Patent ID for logging
            stage_name: Stage name for logging
            max_continuations: Maximum number of continuations (defaults to MAX_CONTINUATIONS)
        
        Returns:
            tuple: (list of TSV responses, list of usage stats)
        """
        if max_continuations is None:
            max_continuations = self.MAX_CONTINUATIONS
        return await request_continuation_basic(
            llm=self.llm,
            messages=messages,
            patent_id=patent_id,
            stage_name=stage_name,
            continuation_prompt=self.CONTINUATION_PROMPT,
            max_continuations=max_continuations,
            logger=self.logger,
        )

    def _filter_stage1_for_stage2(self, stage1_tsv: str) -> str:
        return filter_stage1_for_stage2(stage1_tsv, logger=self.logger)

    def _build_stage0_user_prompt(self, xml_string: str) -> str:
        return f"""--- Full Patent XML Document ---

{xml_string}

---
"""

    def _build_base_context(self, stage0_user_prompt: str) -> List[Dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": stage0_user_prompt},
        ]

    async def _run_stage0(self, patent_id: str, stage0_user_prompt: str):
        self.logger.info(f"Stage 0: Caching patent XML for {patent_id}")
        messages_stage0 = self._build_base_context(stage0_user_prompt)

        stage0_response, stage0_usage = await self.llm.async_call_llm(
            patent_id=patent_id,
            messages=messages_stage0,
            max_tokens=1,
        )

        self.logger.info(
            f"Stage 0 completed: XML cached (response: {len(stage0_response) if stage0_response else 0} chars)"
        )
        return stage0_response, stage0_usage

    async def _run_stage1(self, patent_id: str, base_context: List[Dict]):
        self.logger.info(f"Stage 1: Extracting assay information from patent {patent_id}")
        messages_stage1 = base_context + [
            {"role": "user", "content": self.USER_PROMPT_STAGE1}
        ]

        stage1_response, stage1_usage = await self.llm.async_call_llm(
            patent_id=patent_id,
            messages=messages_stage1,
        )

        if not stage1_response:
            self.logger.error(f"Stage 1 failed for patent {patent_id}")
            raise RuntimeError(f"Stage 1 failure: No response received for patent {patent_id}")

        self.logger.info(f"Stage 1 completed: {len(stage1_response)} chars")
        return stage1_response, stage1_usage

    def _parse_stage1_assays(self, stage1_response: str, patent_id: str) -> List[Dict]:
        stage1_response_list = self._ensure_header_in_tsv(
            [stage1_response], StageFormats.STAGE1_HEADER, "Stage 1", patent_id
        )
        stage1_assays_raw = self._merge_tsv_parts(stage1_response_list, "Stage 1", patent_id)
        return self._deduplicate_stage1_assays(stage1_assays_raw)

    def _deduplicate_stage1_assays(self, stage1_assays_raw: List[Dict]) -> List[Dict]:
        seen_assay_rows = set()
        stage1_assays = []
        duplicate_assay_count = 0

        for item in stage1_assays_raw:
            row_tuple = tuple(
                (item.get(key, "") or "").strip()
                for key in StageFormats.STAGE1_COLUMNS
            )

            if row_tuple in seen_assay_rows:
                duplicate_assay_count += 1
                self.logger.debug(f"Stage 1: Skipping duplicate row: {item}")
            else:
                seen_assay_rows.add(row_tuple)
                stage1_assays.append(item)

        if duplicate_assay_count > 0:
            self.logger.warning(
                f"Stage 1: Removed {duplicate_assay_count} duplicate rows, "
                f"kept {len(stage1_assays)} unique rows from {len(stage1_assays_raw)} total"
            )

        return stage1_assays

    async def _run_stage2(self, patent_id: str, base_context: List[Dict], stage1_response: str):
        self.logger.info(f"Stage 2: Extracting bioactivity data from patent {patent_id}")
        stage1_filtered = self._filter_stage1_for_stage2(stage1_response)
        stage2_prompt = self.USER_PROMPT_STAGE2.format(stage1_data=stage1_filtered)
        messages_stage2 = base_context + [
            {"role": "user", "content": stage2_prompt}
        ]

        stage2_responses, stage2_usage_list, stage2_diagnostics = await self.request_continuation(
            messages=messages_stage2,
            patent_id=patent_id,
            stage_name="Stage 2",
            expected_columns=StageFormats.STAGE2_COLUMN_COUNT,
        )
        stage2_responses = self._ensure_header_in_tsv(
            stage2_responses, StageFormats.STAGE2_HEADER, "Stage 2", patent_id
        )

        self.logger.info(f"Stage 2 [{patent_id}]: completed with {len(stage2_responses)} responses")
        return stage2_responses, stage2_usage_list, stage2_diagnostics

    def _parse_stage2_bioactivity(self, stage2_responses: List[str], patent_id: str) -> List[Dict]:
        stage2_bioactivity_raw = self._merge_tsv_parts(stage2_responses, "Stage 2", patent_id)
        return self._deduplicate_stage2_bioactivity(stage2_bioactivity_raw)

    def _deduplicate_stage2_bioactivity(self, stage2_bioactivity_raw: List[Dict]) -> List[Dict]:
        seen_rows = set()
        stage2_bioactivity = []
        duplicate_count = 0

        for item in stage2_bioactivity_raw:
            dedup_columns = [col for col in StageFormats.STAGE2_COLUMNS if col != 'reasoning']
            row_tuple = tuple(
                (item.get(key, "") or "").strip()
                for key in dedup_columns
            )

            if row_tuple in seen_rows:
                duplicate_count += 1
                self.logger.debug(f"Stage 2: Skipping duplicate row: {item}")
            else:
                seen_rows.add(row_tuple)
                stage2_bioactivity.append(item)

        if duplicate_count > 0:
            self.logger.warning(
                f"Stage 2: Removed {duplicate_count} duplicate rows, "
                f"kept {len(stage2_bioactivity)} unique rows from {len(stage2_bioactivity_raw)} total"
            )

        return stage2_bioactivity

    def _format_stage3_compounds_input(self, stage2_bioactivity: List[Dict], patent_id: str) -> str:
        seen_normalized = {}
        for item in stage2_bioactivity:
            compound = item.get("compound", "").strip()
            if not compound:
                continue
            normalized = re.sub(r'\s+', ' ', compound.lower())
            if normalized not in seen_normalized:
                seen_normalized[normalized] = compound

        unique_compounds = sorted(seen_normalized.values())

        if not unique_compounds:
            self.logger.warning(f"Stage 3: No compounds found in Stage 2 for patent {patent_id}")
            return "[No compounds found in Stage 2]"

        self.logger.info(f"Stage 3 [{patent_id}]: Processing {len(unique_compounds)} unique compounds")
        return "\n".join(unique_compounds)

    async def _run_stage3(self, patent_id: str, base_context: List[Dict], stage2_bioactivity: List[Dict]):
        self.logger.info(f"Stage 3: Extracting final compounds from patent {patent_id}")
        compounds_str = self._format_stage3_compounds_input(stage2_bioactivity, patent_id)
        messages_stage3 = base_context + [
            {"role": "user", "content": self.USER_PROMPT_STAGE3.format(compounds=compounds_str)}
        ]

        stage3_responses, stage3_usage_list, stage3_diagnostics = await self.request_continuation(
            messages=messages_stage3,
            patent_id=patent_id,
            stage_name="Stage 3",
            expected_columns=StageFormats.STAGE3_COLUMN_COUNT,
        )
        stage3_responses = self._ensure_header_in_tsv(
            stage3_responses, StageFormats.STAGE3_HEADER, "Stage 3", patent_id
        )

        self.logger.info(f"Stage 3 [{patent_id}]: completed with {len(stage3_responses)} responses")
        return stage3_responses, stage3_usage_list, stage3_diagnostics

    def _parse_and_clean_stage3_compounds(self, stage3_responses: List[str], patent_id: str) -> List[Dict]:
        stage3_compounds = self._merge_tsv_parts(stage3_responses, "Stage 3", patent_id)

        duplicate_iupac = self._check_duplicate_iupac_names(stage3_compounds)
        if self.REMOVE_DUPLICATE_IUPAC and duplicate_iupac:
            stage3_compounds_cleaned = self._remove_duplicate_iupac_names(stage3_compounds, duplicate_iupac)
        else:
            stage3_compounds_cleaned = stage3_compounds

        duplicate_chem_ids = self._check_duplicate_chemical_ids(stage3_compounds_cleaned)
        if self.REMOVE_DUPLICATE_CHEMICAL_ID and duplicate_chem_ids:
            stage3_compounds_cleaned = self._remove_duplicate_chemical_ids(stage3_compounds_cleaned, duplicate_chem_ids)

        return stage3_compounds_cleaned

    def _prepare_valid_data(
        self,
        stage1_assays: List[Dict],
        stage2_bioactivity: List[Dict],
        stage3_compounds_cleaned: List[Dict],
        doc: PatentDocument,
        xml_string: str,
    ) -> List[Dict]:
        extracted_data = self._merge_with_assay_id(stage1_assays, stage2_bioactivity, stage3_compounds_cleaned, doc)

        valid_data = []
        for item in extracted_data:
            if self._validate_item(item):
                valid_data.append(item)

        self.logger.info(f"Extracted {len(valid_data)} valid data points from {len(extracted_data)} total")

        for item in valid_data:
            if item.get("compound"):
                item["molecule_name"] = item["compound"]
            elif item.get("compound_IUPAC_name"):
                item["molecule_name"] = item["compound_IUPAC_name"]
            else:
                item["molecule_name"] = None

        for item in valid_data:
            item["binding_context"] = f"Full XML (length: {len(xml_string)} chars)"
            item["source_description"] = "Complete patent XML (three-stage extraction)"

        self._enrich_with_smiles(valid_data, doc)
        return valid_data

    def _build_no_xml_result(self) -> Dict:
        return {
            "final_data": [],
            "debug_artifacts": [{
                "source_chunk": "No XML",
                "source_info": "Patent has no XML root",
                "llm_response": None,
                "usage_stats": None,
                "extracted_items": []
            }]
        }

    def _build_stage1_empty_result(self, doc, stage1_response, stage0_usage, stage1_usage) -> Dict:
        usage_stats_list = self._prepare_usage_stats_list(stage0_usage, stage1_usage, [], [])
        return {
            "patent_id": doc.patent_id,
            "extracted_data": [],
            "source_info": f"Complete patent XML - Stage 1 empty, no bioactivity data found",
            "stage1_response": stage1_response,
            "stage2_responses": [],
            "stage3_responses": [],
            "usage_stats": usage_stats_list,
            "stage1_assays": [],
            "stage2_bioactivity": [],
            "stage3_compounds": []
        }

    def _build_stage2_empty_result(
        self,
        doc,
        stage1_response,
        stage2_responses,
        stage0_usage,
        stage1_usage,
        stage2_usage_list,
        stage1_assays,
    ) -> Dict:
        usage_stats_list = self._prepare_usage_stats_list(stage0_usage, stage1_usage, stage2_usage_list, [])
        return {
            "patent_id": doc.patent_id,
            "extracted_data": [],
            "source_info": f"Complete patent XML - Stage 2 empty, no compounds found",
            "stage1_response": stage1_response,
            "stage2_responses": stage2_responses,
            "stage3_responses": [],
            "usage_stats": usage_stats_list,
            "stage1_assays": stage1_assays,
            "stage2_bioactivity": [],
            "stage3_compounds": []
        }

    def _build_final_result(
        self,
        xml_string,
        valid_data,
        stage1_response,
        stage2_responses,
        stage3_responses,
        stage2_diagnostics,
        stage3_diagnostics,
        usage_stats_list,
    ) -> Dict:
        return {
            "final_data": valid_data,
            "debug_artifacts": [{
                "source_chunk": xml_string[:1000] + "..." if len(xml_string) > 1000 else xml_string,
                "source_info": f"Complete patent XML (three-stage with caching, Stage2 bioactivity: {len(stage2_responses)} parts, Stage3 compounds: {len(stage3_responses)} parts)",
                "stage1_response": stage1_response,
                "stage2_responses": stage2_responses,
                "stage3_responses": stage3_responses,
                "stage2_diagnostics": stage2_diagnostics,
                "stage3_diagnostics": stage3_diagnostics,
                "usage_stats_list": usage_stats_list,
                "extracted_items": [{"item": item, "is_valid": True} for item in valid_data]
            }]
        }

    async def async_process_document(self, doc: PatentDocument, output_dir: Optional[str] = None) -> Dict:
        """
        Process the full patent using a four-stage approach with prompt caching.

        Stage 0: Cache patent XML (Cache Write with max_tokens=1, response ignored)
        Stage 1: Extract assay information (Cache Read from Stage 0)
        Stage 2: Extract bioactivities (Cache Read from Stage 0 + Stage 1 data in prompt)
        Stage 3: Extract final compounds (Cache Read from Stage 0 + Stage 2 compounds in prompt)

        ARCHITECTURE (base context is reused):

        base_context = [System, XML]  <- cached in Stage 0

        - Stage 1: base_context + [Stage1_prompt]
        - Stage 2: base_context + [Stage2_prompt.format(stage1_data=Stage1_TSV)]
        - Stage 3: base_context + [Stage3_prompt.format(compounds=Stage2_list)]

        OPTIMIZATION:
        1. All stages use the same base_context (maximum cache hit rate)
        2. Data between stages is passed explicitly via prompt placeholders
        3. No dialog history accumulation -> minimal context size
        4. Stage 0 creates the cache with max_tokens=1 (minimal cache write cost)

        Args:
            doc: PatentDocument with xml_root
            output_dir: Directory for saving results (if None, 'llm_responses' is used)

        Returns:
            dict: {"final_data": [...], "debug_artifacts": [...]}

        Raises:
            RuntimeError: If the LLM returns no response
        """
        if doc.xml_root is None:
            self.logger.warning(f"No XML root for patent {doc.patent_id}")
            return self._build_no_xml_result()

        xml_string = self._extract_relevant_xml_sections(doc.xml_root)
        self.logger.info(f"Processing patent {doc.patent_id}, extracted XML length: {len(xml_string)} characters")

        stage0_user_prompt = self._build_stage0_user_prompt(xml_string)
        _stage0_response, stage0_usage = await self._run_stage0(doc.patent_id, stage0_user_prompt)
        base_context = self._build_base_context(stage0_user_prompt)

        stage1_response, stage1_usage = await self._run_stage1(doc.patent_id, base_context)
        stage1_assays = self._parse_stage1_assays(stage1_response, doc.patent_id)

        if not stage1_assays:
            self.logger.warning(
                f"Stage 1 returned no data for patent {doc.patent_id}. "
                "Skipping Stage 2 and Stage 3."
            )
            self._save_stage_responses(
                doc.patent_id,
                stage1_response,
                [],
                [],
                None,
                output_dir
            )
            return self._build_stage1_empty_result(doc, stage1_response, stage0_usage, stage1_usage)

        stage2_responses, stage2_usage_list, stage2_diagnostics = await self._run_stage2(
            doc.patent_id, base_context, stage1_response
        )
        stage2_bioactivity = self._parse_stage2_bioactivity(stage2_responses, doc.patent_id)

        if not stage2_bioactivity:
            self.logger.warning(
                f"Stage 2 returned no data for patent {doc.patent_id}. "
                "Skipping Stage 3."
            )
            self._save_stage_responses(
                doc.patent_id,
                stage1_response,
                stage2_responses,
                [],
                None,
                output_dir,
                stage2_failed=stage2_diagnostics.get('failed_responses')
            )
            return self._build_stage2_empty_result(
                doc,
                stage1_response,
                stage2_responses,
                stage0_usage,
                stage1_usage,
                stage2_usage_list,
                stage1_assays,
            )

        stage3_responses, stage3_usage_list, stage3_diagnostics = await self._run_stage3(
            doc.patent_id, base_context, stage2_bioactivity
        )
        stage3_compounds_cleaned = self._parse_and_clean_stage3_compounds(stage3_responses, doc.patent_id)
        valid_data = self._prepare_valid_data(
            stage1_assays, stage2_bioactivity, stage3_compounds_cleaned, doc, xml_string
        )

        self._save_stage_responses(
            doc.patent_id, 
            stage1_response, 
            stage2_responses, 
            stage3_responses,
            stage3_compounds_cleaned if (self.REMOVE_DUPLICATE_IUPAC or self.REMOVE_DUPLICATE_CHEMICAL_ID) else None,
            output_dir,
            stage2_failed=stage2_diagnostics.get('failed_responses'),
            stage3_failed=stage3_diagnostics.get('failed_responses')
        )

        usage_stats_list = self._prepare_usage_stats_list(stage0_usage, stage1_usage, stage2_usage_list, stage3_usage_list)

        if stage2_diagnostics:
            self.logger.info(f"Stage 2 diagnostics: {stage2_diagnostics}")
        if stage3_diagnostics:
            self.logger.info(f"Stage 3 diagnostics: {stage3_diagnostics}")

        return self._build_final_result(
            xml_string,
            valid_data,
            stage1_response,
            stage2_responses,
            stage3_responses,
            stage2_diagnostics,
            stage3_diagnostics,
            usage_stats_list,
        )
    
    def _normalize_compound_alias(self, alias: str) -> str:
        return normalize_compound_alias(alias)
    
    def _extract_compound_number(self, alias: str) -> str:
        return extract_compound_number(alias)
    
    def _merge_with_assay_id(
        self, 
        stage1_data: List[Dict],  # Assays with assay_id
        stage2_data: List[Dict],  # Bioactivities with assay_id
        stage3_data: List[Dict],  # Compounds
        doc: PatentDocument
    ) -> List[Dict]:
        return merge_with_assay_id(stage1_data, stage2_data, stage3_data, doc, logger=self.logger)

    def _validate_item(self, item: object) -> bool:
        return validate_item(item)

    def _save_stage_responses(
        self, 
        patent_id: str, 
        stage1_response: str, 
        stage2_responses: List[str], 
        stage3_responses: List[str],
        stage3_compounds_cleaned: Optional[List[Dict]] = None,
        output_dir: Optional[str] = None,
        stage2_failed: Optional[List] = None,
        stage3_failed: Optional[List] = None
    ) -> None:
        return save_stage_responses(
            patent_id,
            stage1_response,
            stage2_responses,
            stage3_responses,
            stage3_compounds_cleaned=stage3_compounds_cleaned,
            output_dir=output_dir,
            stage2_failed=stage2_failed,
            stage3_failed=stage3_failed,
            logger=self.logger,
        )
    
    def _prepare_usage_stats_list(self, stage0_usage: Optional[Dict], stage1_usage: Optional[Dict], stage2_usage_list: List[Dict], stage3_usage_list: List[Dict]) -> List[Dict]:
        return prepare_usage_stats_list(
            stage0_usage,
            stage1_usage,
            stage2_usage_list,
            stage3_usage_list,
            logger=self.logger,
        )
    
    def _enrich_with_smiles(self, valid_data: List[Dict], doc: PatentDocument) -> None:
        return enrich_with_smiles(
            valid_data,
            doc,
            PY2OPSIN_AVAILABLE,
            py2opsin if PY2OPSIN_AVAILABLE else None,
            logger=self.logger,
        )

