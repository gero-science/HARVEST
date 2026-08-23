def validate_item(item: object) -> bool:
    """Validate a merged extraction item using the historical minimum fields."""
    if not isinstance(item, dict):
        return False

    has_identifier = (
        item.get("compound") or
        item.get("compound_IUPAC_name") or
        item.get("chemical_id")
    )
    if not has_identifier:
        return False

    if not (item.get("value")
            and item.get("binding_metric")
            and item.get("protein_target_name")):
        return False

    return True
