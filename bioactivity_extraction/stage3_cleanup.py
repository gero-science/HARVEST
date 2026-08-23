from typing import Dict, List


def check_duplicate_iupac_names(compounds_data: List[Dict], logger=None) -> set:
    if not compounds_data:
        return set()

    iupac_frequency = {}
    for item in compounds_data:
        iupac_name = (item.get("compound_IUPAC_name") or "").strip()
        if iupac_name:
            iupac_frequency[iupac_name] = iupac_frequency.get(iupac_name, 0) + 1

    duplicate_iupac_names = {name for name, count in iupac_frequency.items() if count > 1}

    if duplicate_iupac_names and logger:
        total_entries = sum(iupac_frequency[name] for name in duplicate_iupac_names)
        logger.warning(
            f"Stage 3: Found {len(duplicate_iupac_names)} duplicate IUPAC names "
            f"affecting {total_entries} compound entries"
        )
        for dup_name in list(duplicate_iupac_names)[:5]:
            count = iupac_frequency[dup_name]
            logger.warning(f"  - '{dup_name[:80]}...' (appears {count} times)")
        if len(duplicate_iupac_names) > 5:
            logger.warning(f"  ... and {len(duplicate_iupac_names) - 5} more")

    return duplicate_iupac_names


def remove_duplicate_iupac_names(compounds_data: List[Dict], duplicate_names: set) -> List[Dict]:
    cleaned_data = []
    for item in compounds_data:
        item_copy = item.copy()
        iupac_name = (item_copy.get("compound_IUPAC_name") or "").strip()
        if iupac_name in duplicate_names:
            item_copy["compound_IUPAC_name"] = ""
        cleaned_data.append(item_copy)
    return cleaned_data


def check_duplicate_chemical_ids(compounds_data: List[Dict], logger=None) -> set:
    if not compounds_data:
        return set()

    chemical_id_frequency = {}
    for item in compounds_data:
        chem_id = (item.get("chemical_id") or "").strip()
        if chem_id:
            chemical_id_frequency[chem_id] = chemical_id_frequency.get(chem_id, 0) + 1

    duplicate_chemical_ids = {cid for cid, count in chemical_id_frequency.items() if count > 1}

    if duplicate_chemical_ids and logger:
        total_entries = sum(chemical_id_frequency[cid] for cid in duplicate_chemical_ids)
        logger.warning(
            f"Stage 3: Found {len(duplicate_chemical_ids)} duplicate chemical_id "
            f"affecting {total_entries} compound entries"
        )
        for dup_cid in list(duplicate_chemical_ids)[:5]:
            count = chemical_id_frequency[dup_cid]
            logger.warning(f"  - '{dup_cid}' (appears {count} times)")
        if len(duplicate_chemical_ids) > 5:
            logger.warning(f"  ... and {len(duplicate_chemical_ids) - 5} more")

    return duplicate_chemical_ids


def remove_duplicate_chemical_ids(compounds_data: List[Dict], duplicate_chem_ids: set) -> List[Dict]:
    cleaned_data = []
    for item in compounds_data:
        item_copy = item.copy()
        chem_id = (item_copy.get("chemical_id") or "").strip()
        if chem_id in duplicate_chem_ids:
            item_copy["chemical_id"] = ""
        cleaned_data.append(item_copy)
    return cleaned_data
