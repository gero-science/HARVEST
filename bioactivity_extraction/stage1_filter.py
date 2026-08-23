def filter_stage1_for_stage2(stage1_tsv: str, logger=None) -> str:
    """Filter Stage 1 TSV to the columns passed into Stage 2."""
    if not stage1_tsv or not stage1_tsv.strip():
        return ""

    needed_columns = [
        "assay_id",
        "protein_target_name",
        "protein_modification",
        "assay_description",
        "organism"
    ]

    try:
        lines = stage1_tsv.strip().split('\n')
        if not lines:
            return ""

        header_line = lines[0]
        header_cols = header_line.split('\t')

        col_indices = []
        for needed_col in needed_columns:
            try:
                idx = header_cols.index(needed_col)
                col_indices.append(idx)
            except ValueError:
                if logger:
                    logger.warning(f"Column '{needed_col}' not found in Stage 1 header")
                col_indices.append(-1)

        filtered_lines = ['\t'.join(needed_columns)]
        seen_rows = set()
        duplicate_count = 0

        for line in lines[1:]:
            if not line.strip():
                continue

            cols = line.split('\t')
            filtered_cols = []
            for idx in col_indices:
                if idx >= 0 and idx < len(cols):
                    filtered_cols.append(cols[idx])
                else:
                    filtered_cols.append("")

            filtered_line = '\t'.join(filtered_cols)
            row_tuple = tuple(filtered_cols)

            if row_tuple in seen_rows:
                duplicate_count += 1
                if logger:
                    logger.debug(f"Skipping duplicate row after filtering: {filtered_line[:100]}")
                continue

            seen_rows.add(row_tuple)
            filtered_lines.append(filtered_line)

        filtered_tsv = '\n'.join(filtered_lines)
        original_rows = len(lines) - 1
        filtered_rows = len(filtered_lines) - 1

        if logger:
            log_msg = (
                f"Filtered Stage 1 data for Stage 2: {original_rows} rows → {filtered_rows} unique rows, "
                f"{len(header_cols)} → {len(needed_columns)} columns"
            )
            if duplicate_count > 0:
                log_msg += f", removed {duplicate_count} duplicates"
            logger.info(log_msg)

        return filtered_tsv

    except Exception as e:
        if logger:
            logger.error(f"Failed to filter Stage 1 TSV: {e}")
        return stage1_tsv
