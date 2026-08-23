import csv
import logging
from io import StringIO
from typing import Dict, List, Set, Tuple

from .prompts import load_prompt_file
from .stage_formats import StageFormats


class ImprovedContinuationLogic:
    """Improved continuation logic with loop protection."""
    
    # Configuration constants (can be overridden by subclassing)
    MAX_CONTINUATIONS = 3           # Maximum continuation requests
    MAX_DUPLICATE_RATIO = 0.05      # Stop if >5% duplicates detected
    MIN_NEW_ROWS = 8               # Minimum unique rows to continue
    MAX_TOKENS_RESPONSE = 65536     # Gemini 2.5 Flash max output tokens

    # AGGRESSIVE anti-looping thresholds
    FIRST_CONTINUATION_MAX_DUP_RATIO = 0.20  # First continuation: stop if >20% duplicates
    MIN_UNIQUE_RATIO = 0.30          # Stop if <30% unique rows (i.e. >70% duplicates)

    MASS_DUP_MIN_REPEAT_COUNT = 2    # Rows repeated 2+ times
    MASS_DUP_MIN_AFFECTED_ROWS = 5   # At least 5 different rows
    
    COMPLETION_SIGNAL = "[NO_DATA]"  # Completion signal (filtered from output)
    
    # Improved prompt with loop protection
    CONTINUATION_PROMPT = load_prompt_file('impr_continuation_prompt.txt')
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize."""
        self.logger = logger or logging.getLogger(__name__)
        self.known_headers: Set[tuple] = set()  # Dynamically learned headers
    
    def is_response_truncated(
        self, 
        response_text: str, 
        completion_tokens: int, 
        expected_columns: int
    ) -> Tuple[bool, str]:
        """
        Check whether the response was truncated using multiple signals.
        
        Content-based checks only apply when response is near token limit (>95%)
        to avoid false positives on valid responses. Below 95%, we trust finish_reason.
        
        Args:
            response_text: Response text
            completion_tokens: Number of tokens in the completion
            expected_columns: Expected number of TSV columns
        
        Returns:
            (is_truncated, reason)
        """
        # 1. Check token usage
        if completion_tokens <= 0:
            return False, ""  # No token info, trust finish_reason
        
        token_ratio = completion_tokens / self.MAX_TOKENS_RESPONSE
        
        # If far from limit (<95%), trust response is complete
        # Only apply content checks when very close to token limit
        if token_ratio < 0.95:
            return False, ""
        
        # At >95% of limit, apply content-based checks
        near_limit = True
        
        if not response_text or not response_text.strip():
            return False, ""
        
        lines = response_text.strip().split('\n')
        if not lines:
            return False, ""
        
        last_line = lines[-1].strip()
        if not last_line and len(lines) > 1:
            last_line = lines[-2].strip()
        
        if not last_line:
            return False, ""
        
        # Content-based checks (only when >95% tokens used)
        
        # 2. Check if last line has wrong column count (incomplete row)
        if '\t' in last_line:
            last_cols = len(last_line.split('\t'))
            if last_cols != expected_columns:
                return True, f"Near limit: last line has {last_cols} cols (expected {expected_columns})"
        
        # 3. Check if response doesn't end with newline
        if not response_text.endswith('\n'):
            return True, "Near max tokens and no trailing newline"
        
        # Check 4 REMOVED (too aggressive - was flagging valid long cells)
        
        return False, ""
    
    def parse_tsv_to_rows(self, tsv_text: str, skip_header: bool = True) -> Tuple[List[tuple], int]:
        """
        Parse TSV text into a list of tuples (for comparison).
        
        Rows with the wrong number of columns are skipped automatically.
        
        Args:
            tsv_text: TSV text to parse
            skip_header: Skip the first row as a header (True for initial response, False for continuations)
        
        Returns:
            (list_of_row_tuples, expected_columns)
        """
        if not tsv_text or not tsv_text.strip():
            return [], 0
        
        rows = []
        expected_cols = 0
        skipped_wrong_cols = 0
        
        # Known header patterns to filter out (in case LLM repeats them in continuations)
        # Use centralized patterns from stage_formats.py
        HEADER_PATTERNS = StageFormats.ALL_HEADER_PATTERNS
        
        try:
            reader = csv.reader(StringIO(tsv_text.strip()), delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    expected_cols = len(row)
                    if skip_header:
                        # Store normalized header for future filtering
                        normalized_header = tuple(cell.lower().replace('_', '').replace('-', '') for cell in row)
                        self.known_headers.add(normalized_header)
                        # Skip header only if requested (initial response)
                        continue
                
                # Skip rows with the wrong number of columns
                if expected_cols > 0 and len(row) != expected_cols:
                    skipped_wrong_cols += 1
                    self.logger.debug(
                        f"Skipping row {i+1}: {len(row)} columns (expected {expected_cols})"
                    )
                    continue
                
                # Convert to tuple for hashing
                row_tuple = tuple(cell.strip() if cell else '' for cell in row)
                
                # Filter out invalid rows (headers, completion signals, empty)
                if self._is_invalid_row(row_tuple, HEADER_PATTERNS):
                    continue
                
                rows.append(row_tuple)
            
            if skipped_wrong_cols > 0:
                self.logger.info(
                    f"Parsed {len(rows)} valid rows, skipped {skipped_wrong_cols} rows with wrong column count"
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to parse TSV: {e}")
            return [], 0
        
        return rows, expected_cols
    
    def _is_invalid_row(self, row: tuple, header_patterns: List[tuple]) -> bool:
        """
        Check whether a row is invalid (header, NO_MORE_DATA, empty).
        
        Args:
            row: Tuple of row values
            header_patterns: List of known header patterns
        
        Returns:
            True if the row is invalid and should be filtered out
        """
        # Empty row
        if not row or all(not cell for cell in row):
            return True
        
        # Check for completion signal (exact match in any cell)
        for cell in row:
            if cell.strip() == self.COMPLETION_SIGNAL:
                return True
        
        # Check if row matches header pattern
        normalized_row = tuple(cell.lower().replace('_', '').replace('-', '') for cell in row)
        
        # Check against dynamically learned headers first (priority)
        for known_header in self.known_headers:
            if len(normalized_row) >= len(known_header):
                if normalized_row[:len(known_header)] == known_header:
                    return True
        
        # Fallback to hardcoded patterns (for backward compatibility)
        for pattern in header_patterns:
            normalized_pattern = tuple(cell.lower().replace('_', '').replace('-', '') for cell in pattern)
            # Check if row starts with pattern (allows partial matches)
            if len(normalized_row) >= len(normalized_pattern):
                if normalized_row[:len(normalized_pattern)] == normalized_pattern:
                    return True
        
        return False
    
    def detect_duplicates(
        self,
        previous_rows: Set[tuple],
        new_rows: List[tuple]
    ) -> Tuple[int, int, float, dict]:
        """
        Count duplicates in new rows and detect mass duplication.

        LOGIC:
        - Duplicates = ALL records that appear more than once (within new_rows OR in previous_rows)
        - Count the SUM of all repeated records (not unique records, but all copies)
        - Duplicate ratio = sum of all repeats / total rows in new_rows
        - If duplicate_ratio > 10%, continuation should stop

        Returns:
            (unique_count, duplicate_count, duplicate_ratio, mass_duplication_info)

        mass_duplication_info contains:
            - has_mass_duplication: bool - whether mass duplication was detected
            - repeat_count: int - how many times records repeat
            - affected_rows: int - number of distinct records with the same repeat_count
            - description: str - problem description
        """
        if not new_rows:
            return 0, 0, 0.0, {'has_mass_duplication': False}

        # Count how many times each row appears in new_rows
        row_counts = {}
        for row in new_rows:
            if row in row_counts:
                row_counts[row] += 1
            else:
                row_counts[row] = 1

        # duplicate_count = sum of ALL records that appear more than once
        unique_count = 0      # Records that appear exactly once AND are not in previous_rows
        duplicate_count = 0   # SUM of all repeated records

        for row, count in row_counts.items():
            # If the row was already in previous_rows, all occurrences are duplicates
            if row in previous_rows:
                duplicate_count += count
            # If the row appears more than once in new_rows, all occurrences are duplicates
            elif count > 1:
                duplicate_count += count
            # If the row appears exactly once and is not in previous_rows, it is unique
            else:
                unique_count += 1

        # Duplicate ratio = share of duplicates among all rows
        duplicate_ratio = duplicate_count / len(new_rows) if new_rows else 0.0

        # Mass duplication detection:
        # If many different rows repeat the same number of times (>= 3),
        # that is a sign the LLM is outputting the same block repeatedly

        # Group rows by repeat count
        repeat_groups = {}  # repeat_count -> list of rows
        for row, count in row_counts.items():
            if count >= 2:  # Only interested in repeated rows
                if count not in repeat_groups:
                    repeat_groups[count] = []
                repeat_groups[count].append(row)

        mass_duplication_info = {'has_mass_duplication': False}

        # Check whether there is a group where >= 5 distinct rows repeat >= 2 times (AGGRESSIVE)
        for repeat_count, rows_with_this_count in repeat_groups.items():
            if repeat_count >= self.MASS_DUP_MIN_REPEAT_COUNT and len(rows_with_this_count) >= self.MASS_DUP_MIN_AFFECTED_ROWS:
                mass_duplication_info = {
                    'has_mass_duplication': True,
                    'repeat_count': repeat_count,
                    'affected_rows': len(rows_with_this_count),
                    'description': f"{len(rows_with_this_count)} different rows repeated {repeat_count} times each"
                }
                break  # Found the first problematic group

        return unique_count, duplicate_count, duplicate_ratio, mass_duplication_info
    
    def validate_tsv_format(
        self, 
        tsv_text: str, 
        expected_columns: int
    ) -> Tuple[bool, str]:
        """
        Validate the TSV response format.
        
        Rows with the wrong number of columns are skipped rather than treated as errors.
        Validation succeeds if there is at least one valid row.
        
        Returns:
            (is_valid, error_message)
        """
        if not tsv_text or not tsv_text.strip():
            return False, "Empty response"
        
        # Check for completion signal (exact match on its own line)
        lines = tsv_text.strip().split('\n')
        for line in lines:
            if line.strip() == self.COMPLETION_SIGNAL:
                return True, "Completion signal received"
        
        try:
            lines = tsv_text.strip().split('\n')
            
            valid_line_count = 0
            invalid_line_count = 0
            
            # Check each data line (headers already filtered by _is_invalid_row)
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                cols = len(line.split('\t'))
                if cols == expected_columns:
                    valid_line_count += 1
                else:
                    invalid_line_count += 1
                    self.logger.debug(
                        f"Skipping line {i+1}: {cols} columns (expected {expected_columns}), "
                        f"content: {line[:100]}..."
                    )
            
            # Treat as valid if there is at least one correct row
            if valid_line_count > 0:
                if invalid_line_count > 0:
                    self.logger.info(
                        f"Validated format: {valid_line_count} valid lines, "
                        f"{invalid_line_count} lines skipped (wrong column count)"
                    )
                return True, "Valid format"
            else:
                return False, f"No valid lines found (all {invalid_line_count} lines have wrong column count)"
            
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def parse_partial_tsv(
        self, 
        tsv_text: str, 
        expected_columns: int
    ) -> Tuple[List[str], List[str]]:
        """
        Parse TSV text, separating valid and invalid rows.
        
        Used to extract data from partially corrupted responses.
        
        Args:
            tsv_text: TSV text to parse
            expected_columns: Expected number of columns
        
        Returns:
            (valid_lines, invalid_lines) - lists of text lines
        """
        if not tsv_text or not tsv_text.strip():
            return [], []
        
        valid_lines = []
        invalid_lines = []
        
        # Known header patterns to filter out
        # Use centralized patterns from stage_formats.py
        HEADER_PATTERNS = StageFormats.ALL_HEADER_PATTERNS
        
        lines = tsv_text.strip().split('\n')

        # Track seen rows for deduplication
        seen_rows = set()
        duplicates_count = 0

        for line in lines:
            if not line.strip():
                continue

            # Skip completion signal
            if line.strip() == self.COMPLETION_SIGNAL:
                continue

            # Check if line has correct column count
            if '\t' in line:
                cols = line.split('\t')
                if len(cols) == expected_columns:
                    # Convert to tuple for validation and deduplication
                    row_tuple = tuple(cell.strip() if cell else '' for cell in cols)

                    # Filter out invalid rows
                    if not self._is_invalid_row(row_tuple, HEADER_PATTERNS):
                        # Deduplicate: only add if not seen before
                        if row_tuple not in seen_rows:
                            valid_lines.append(line)
                            seen_rows.add(row_tuple)
                        else:
                            # Duplicate row detected
                            duplicates_count += 1
                    else:
                        invalid_lines.append(line)
                else:
                    invalid_lines.append(line)
            else:
                # No tabs - invalid TSV line
                invalid_lines.append(line)

        # Filter out completion signal from valid lines (safety check)
        valid_lines = [line for line in valid_lines if line.strip() != self.COMPLETION_SIGNAL]

        log_msg = f"Partial TSV parse: {len(valid_lines)} unique valid lines"
        if duplicates_count > 0:
            log_msg += f", {duplicates_count} duplicates filtered"
        log_msg += f", {len(invalid_lines)} invalid lines"
        self.logger.info(log_msg)
        
        return valid_lines, invalid_lines
    
    def should_stop_continuation(
        self,
        continuation_count: int,
        unique_count: int,
        duplicate_ratio: float,
        response: str,
        mass_duplication_info: dict = None
    ) -> Tuple[bool, str]:
        """
        Determine whether continuation should stop.

        Args:
            continuation_count: Current continuation number
            unique_count: Number of unique rows in the current response
            duplicate_ratio: Share of duplicates in the current response
            response: Response text
            mass_duplication_info: Mass duplication information

        Returns:
            (should_stop, reason)
        """
        # 1. Check for completion signal (exact match on its own line)
        lines = response.strip().split('\n')
        for line in lines:
            if line.strip() == self.COMPLETION_SIGNAL:
                return True, "LLM signaled completion"

        # 2. Check max continuations
        if continuation_count >= self.MAX_CONTINUATIONS:
            return True, f"Reached max continuations ({self.MAX_CONTINUATIONS})"

        # 3. Check for MASS DUPLICATION (high priority!)
        # If many rows repeat the same number of times, the LLM is looping
        if mass_duplication_info and mass_duplication_info.get('has_mass_duplication'):
            desc = mass_duplication_info.get('description', '')
            return True, f"MASS DUPLICATION detected: {desc} - LLM is looping same block repeatedly"

        # 4. Check for too many duplicates (looping!)
        if duplicate_ratio > self.MAX_DUPLICATE_RATIO:
            return True, f"High duplicate ratio ({duplicate_ratio:.1%}) - likely looping"

        # 5. AGGRESSIVE CHECK: First continuation with high duplicates (>20%)
        # If first continuation already has many duplicates, LLM is repeating data
        if continuation_count == 1 and duplicate_ratio > self.FIRST_CONTINUATION_MAX_DUP_RATIO:
            return True, f"CRITICAL: First continuation has {duplicate_ratio:.1%} duplicates (>{self.FIRST_CONTINUATION_MAX_DUP_RATIO:.0%}) - severe looping detected"

        # 6. CHECK: Too few unique rows relative to total (< 30% unique = > 70% duplicates)
        if len(response.strip().split('\n')) > 0:
            # Count total non-empty lines in response
            total_lines = sum(1 for line in response.strip().split('\n') if line.strip())
            if total_lines > 0:
                unique_ratio = unique_count / total_lines
                if unique_ratio < self.MIN_UNIQUE_RATIO:
                    return True, f"Too few unique rows: {unique_ratio:.1%} unique (< {self.MIN_UNIQUE_RATIO:.0%}) - severe looping"

        # 7. Check for empty continuation (ZERO unique rows)
        # If continuation returned 0 unique rows, that is a clear sign of looping or completion
        if unique_count == 0:
            # Check whether this is just the [NO_DATA] marker
            if self.COMPLETION_SIGNAL not in response:
                return True, "EMPTY CONTINUATION: Zero unique rows but no completion signal - likely severe looping"
            else:
                return True, "Completion signal received (with zero unique rows)"

        # 8. Check for too few new rows (applies to ALL continuations)
        # If continuation returned fewer than MIN_NEW_ROWS unique rows, the LLM is likely exhausted
        if unique_count < self.MIN_NEW_ROWS:
            return True, f"Too few new unique rows ({unique_count} < {self.MIN_NEW_ROWS}) - likely exhausted or hallucinating"

        return False, ""
    
    async def request_continuation_safe(
        self,
        llm,
        messages: List[Dict],
        patent_id: str,
        stage_name: str,
        expected_columns: int
    ) -> Tuple[List[str], List[Dict], Dict]:
        """
        Safe version of request_continuation with loop protection.
        
        Returns:
            (responses, usage_stats, diagnostics)
        """
        responses = []
        usage_stats = []
        all_seen_rows: Set[tuple] = set()
        
        diagnostics = {
            'total_unique_rows': 0,
            'total_duplicate_rows': 0,
            'stopped_early': False,
            'stop_reason': None,
            'continuations': [],
            'failed_responses': [],
            'valid_lines_from_failed': 0
        }
        
        # Initial request
        self.logger.info(f"{stage_name}: making initial request for patent {patent_id}")
        response, usage = await llm.async_call_llm(
            patent_id=patent_id,
            messages=messages.copy()
        )
        
        if not response:
            raise RuntimeError(f"{stage_name}: No response received")
        
        # Validate initial response format
        is_valid, validation_msg = self.validate_tsv_format(response, expected_columns)
        if not is_valid:
            self.logger.warning(f"{stage_name} [{patent_id}]: initial response invalid: {validation_msg}, attempting salvage")
            # Try to salvage valid lines from malformed initial response
            valid_lines, invalid_lines = self.parse_partial_tsv(response, expected_columns)
            if valid_lines:
                response = '\n'.join(valid_lines) + '\n'
                diagnostics['valid_lines_from_failed'] += len(valid_lines)
                self.logger.info(f"{stage_name} [{patent_id}]: salvaged {len(valid_lines)} lines from initial response")
            else:
                self.logger.error(f"{stage_name}: initial response has no valid lines")
        
        # Parse initial response
        initial_rows, cols = self.parse_tsv_to_rows(response)
        if expected_columns == 0:
            expected_columns = cols

        # Check for internal duplicates in initial response
        # This detects if LLM is already looping on first response
        initial_rows_unique = []
        initial_seen = set()
        initial_duplicates = 0

        for row in initial_rows:
            if row in initial_seen:
                initial_duplicates += 1
            else:
                initial_rows_unique.append(row)
                initial_seen.add(row)

        initial_duplicate_ratio = initial_duplicates / len(initial_rows) if initial_rows else 0.0

        all_seen_rows.update(initial_seen)  # Use deduplicated set

        # 🔥 CRITICAL: Rebuild response from ONLY unique rows (remove internal duplicates)
        # Note: parse_tsv_to_rows already skipped header, so initial_rows_unique contains only data rows
        if initial_rows_unique:
            # Extract header from original response (first line)
            original_lines = response.strip().split('\n')
            header_line = original_lines[0] if original_lines else ''

            # Convert unique rows back to TSV
            unique_tsv_lines = ['\t'.join(row) for row in initial_rows_unique]
            # Rebuild with header + unique data rows
            response_deduplicated = header_line + '\n' + '\n'.join(unique_tsv_lines) + '\n'
        else:
            # No data rows - keep only header if present
            original_lines = response.strip().split('\n')
            response_deduplicated = (original_lines[0] + '\n') if original_lines else ''

        responses.append(response_deduplicated)  # Save deduplicated version
        if usage:
            usage_stats.append(usage)

        diagnostics['total_unique_rows'] = len(initial_rows_unique)
        diagnostics['continuations'].append({
            'num': 0,
            'unique': len(initial_rows_unique),
            'duplicates': initial_duplicates,
            'ratio': initial_duplicate_ratio
        })

        finish_reason = usage.get("finish_reason", "stop") if usage else "stop"

        if initial_duplicates > 0:
            self.logger.warning(
                f"{stage_name} [{patent_id}]: initial response - {len(initial_rows_unique)} unique rows, "
                f"{initial_duplicates} internal duplicates ({initial_duplicate_ratio:.1%}), "
                f"finish_reason={finish_reason}"
            )
        else:
            self.logger.info(
                f"{stage_name} [{patent_id}]: initial response - {len(initial_rows_unique)} rows, "
                f"finish_reason={finish_reason}"
            )

        # Continuation loop
        continuation_count = 0
        current_messages = messages.copy()

        # Check if response is truncated (by finish_reason or content-based detection)
        completion_tokens = usage.get("completion_tokens", 0) if usage else 0
        is_truncated, truncation_reason = self.is_response_truncated(response, completion_tokens, expected_columns)

        # DO NOT request continuation if initial response has too many duplicates
        # This indicates LLM is already looping/hallucinating on first response
        if initial_duplicate_ratio > self.MAX_DUPLICATE_RATIO:
            should_continue = False
            self.logger.warning(
                f"{stage_name} [{patent_id}]: NOT requesting continuation - initial response has "
                f"{initial_duplicate_ratio:.1%} internal duplicates (>{self.MAX_DUPLICATE_RATIO:.0%})"
            )
            diagnostics['stopped_early'] = True
            diagnostics['stop_reason'] = f"Initial response has {initial_duplicate_ratio:.1%} internal duplicates"
        else:
            should_continue = (finish_reason == "length" or is_truncated)

        if is_truncated and finish_reason != "length":
            self.logger.info(f"{stage_name} [{patent_id}]: detected truncation via content analysis: {truncation_reason}")
        
        while should_continue and continuation_count < self.MAX_CONTINUATIONS:
            continuation_count += 1
            
            self.logger.info(
                f"{stage_name} [{patent_id}]: requesting continuation {continuation_count}/"
                f"{self.MAX_CONTINUATIONS}"
            )
            
            # Build continuation messages
            current_messages.append({"role": "assistant", "content": response})
            current_messages.append({"role": "user", "content": self.CONTINUATION_PROMPT})
            
            # Request continuation
            response, usage = await llm.async_call_llm(
                patent_id=patent_id,
                messages=current_messages
            )
            
            if not response:
                self.logger.warning(f"{stage_name} [{patent_id}]: continuation {continuation_count} failed")
                break
            
            # Validate format
            is_valid, validation_msg = self.validate_tsv_format(response, expected_columns)
            if not is_valid:
                # Check whether validation_msg indicates the wrong number of columns
                if "wrong column count" in validation_msg.lower() or "no valid lines found" in validation_msg.lower():
                    # Response contains rows with the wrong number of columns
                    # Assume data is complete; do not append the response
                    self.logger.info(
                        f"{stage_name} [{patent_id}]: continuation {continuation_count} has wrong column count, "
                        f"assuming data is complete. Stopping continuation."
                    )
                    diagnostics['stopped_early'] = True
                    diagnostics['stop_reason'] = f"Wrong column count detected: {validation_msg}"
                    diagnostics['failed_responses'].append({
                        'continuation_num': continuation_count,
                        'full_response': response[:500] + "..." if len(response) > 500 else response,
                        'reason': validation_msg
                    })
                    break
                
                self.logger.warning(
                    f"{stage_name} [{patent_id}]: continuation {continuation_count} has invalid format: "
                    f"{validation_msg}, attempting partial parse"
                )
                
                # Try to extract valid lines from malformed response
                valid_lines, invalid_lines = self.parse_partial_tsv(response, expected_columns)
                
                if valid_lines:
                    # Reconstruct response from valid lines only (with trailing newline to mark as complete)
                    response = '\n'.join(valid_lines) + '\n'
                    diagnostics['valid_lines_from_failed'] += len(valid_lines)
                    
                    # Save failed response parts
                    if invalid_lines:
                        diagnostics['failed_responses'].append({
                            'continuation_num': continuation_count,
                            'invalid_lines': invalid_lines,
                            'reason': validation_msg
                        })
                    
                    self.logger.info(
                        f"{stage_name} [{patent_id}]: salvaged {len(valid_lines)} valid lines from "
                        f"malformed response, {len(invalid_lines)} lines failed"
                    )
                else:
                    # No valid lines - complete failure
                    self.logger.error(
                        f"{stage_name} [{patent_id}]: continuation {continuation_count} has no valid lines"
                    )
                    diagnostics['stopped_early'] = True
                    diagnostics['stop_reason'] = f"Invalid format: {validation_msg}"
                    diagnostics['failed_responses'].append({
                        'continuation_num': continuation_count,
                        'full_response': response,
                        'reason': validation_msg
                    })
                    break
            
            # Parse new response (don't skip header in continuations - there is no header!)
            new_rows, _ = self.parse_tsv_to_rows(response, skip_header=False)

            # Check for duplicates (including mass duplication detection)
            unique_count, duplicate_count, duplicate_ratio, mass_duplication_info = self.detect_duplicates(
                all_seen_rows, new_rows
            )

            # 🔥 CRITICAL: Filter out duplicate rows BEFORE adding to responses
            # Keep ONLY rows that are:
            # 1. NOT in all_seen_rows (not from previous responses)
            # 2. First occurrence within new_rows (deduplicate within current response)

            unique_new_rows = []
            seen_in_current = set()

            for row in new_rows:
                # Skip if already seen in previous responses
                if row in all_seen_rows:
                    continue
                # Skip if already encountered in current response
                if row in seen_in_current:
                    continue
                # This is a unique row - keep it
                unique_new_rows.append(row)
                seen_in_current.add(row)

            # Rebuild response from ONLY unique rows
            if unique_new_rows:
                # Convert tuples back to TSV format
                unique_tsv_lines = ['\t'.join(row) for row in unique_new_rows]
                response_deduplicated = '\n'.join(unique_tsv_lines) + '\n'
            else:
                response_deduplicated = ''

            # Log continuation results with mass duplication info
            log_msg = (
                f"{stage_name} [{patent_id}]: continuation {continuation_count} - "
                f"{unique_count} unique, {duplicate_count} duplicates "
                f"({duplicate_ratio:.1%} dup ratio), "
                f"keeping {len(unique_new_rows)} unique rows"
            )
            if mass_duplication_info and mass_duplication_info.get('has_mass_duplication'):
                log_msg += f" [MASS DUP: {mass_duplication_info.get('description', 'detected')}]"
                self.logger.warning(log_msg)
            else:
                self.logger.info(log_msg)

            diagnostics['continuations'].append({
                'num': continuation_count,
                'unique': unique_count,
                'duplicates': duplicate_count,
                'ratio': duplicate_ratio
            })

            # Check if we should stop
            should_stop, stop_reason = self.should_stop_continuation(
                continuation_count, unique_count, duplicate_ratio, response, mass_duplication_info
            )

            if should_stop:
                self.logger.warning(
                    f"{stage_name} [{patent_id}]: stopping continuation early - {stop_reason}"
                )
                diagnostics['stopped_early'] = True
                diagnostics['stop_reason'] = stop_reason

                # Add deduplicated response if it has content and isn't too duplicated
                if duplicate_ratio < self.MAX_DUPLICATE_RATIO and response_deduplicated.strip():
                    responses.append(response_deduplicated)
                    if usage:
                        usage_stats.append(usage)
                    # Update seen rows with ONLY the unique rows we're keeping
                    all_seen_rows.update(unique_new_rows)
                    # Increment diagnostics only when response is included
                    diagnostics['total_unique_rows'] += len(unique_new_rows)
                    diagnostics['total_duplicate_rows'] += duplicate_count

                break

            # Add deduplicated response to responses list
            if response_deduplicated.strip():
                responses.append(response_deduplicated)
            if usage:
                usage_stats.append(usage)
            # Update seen rows with ONLY the unique rows we're keeping
            all_seen_rows.update(unique_new_rows)

            # Increment diagnostics after response is confirmed to be included
            diagnostics['total_unique_rows'] += len(unique_new_rows)
            diagnostics['total_duplicate_rows'] += duplicate_count
            
            # Update continuation condition for next iteration
            finish_reason = usage.get("finish_reason", "stop") if usage else "stop"
            completion_tokens = usage.get("completion_tokens", 0) if usage else 0
            is_truncated, truncation_reason = self.is_response_truncated(response, completion_tokens, expected_columns)
            should_continue = (finish_reason == "length" or is_truncated)
            
            if is_truncated and finish_reason != "length":
                self.logger.info(f"{stage_name} [{patent_id}]: detected truncation via content analysis: {truncation_reason}")
        
        # Log final diagnostics
        if diagnostics.get('stopped_early'):
            self.logger.warning(
                f"{stage_name} [{patent_id}]: Stopped early - {diagnostics['stop_reason']}, "
                f"unique={diagnostics['total_unique_rows']}, "
                f"duplicates={diagnostics['total_duplicate_rows']}"
            )
        else:
            self.logger.info(
                f"{stage_name} [{patent_id}]: Completed normally with {len(responses)} responses, "
                f"unique={diagnostics['total_unique_rows']}, "
                f"duplicates={diagnostics['total_duplicate_rows']}"
            )
        
        # Update original messages list in-place for prompt caching
        # Add all continuation messages from current_messages that aren't in original messages
        messages_to_add = current_messages[len(messages):]
        messages.extend(messages_to_add)
        
        return responses, usage_stats, diagnostics
