import json
import os
import re
import logging
from glob import glob
from pathlib import Path


def _extract_value_and_unit(s: str) -> tuple:
    """
    Extract numeric value and unit from a string.

    Args:
        s: string like "100 nM", ">100nM", "10 µM", "10uM", "1.0E-08", etc.

    Returns:
        tuple: (value: float, unit: str or None)
    """
    s = s.strip().replace("−", "-")

    # Remove leading comparison operators
    s = re.sub(r'^[<>=≤≥≦≧]+\s*', '', s)

    # Scientific notation: 1.0E-08, 2.00e-5, optional unit
    sci = re.match(
        r'^(\d+\.?\d*[eE][+\-]?\d+)\s*([a-zA-Zµμ]+(?:/[a-zA-Z]+)?)?$',
        s,
    )
    if sci:
        try:
            value = float(sci.group(1))
            unit = sci.group(2) if sci.group(2) else None
            return (value, unit)
        except ValueError:
            pass

    # Pattern to extract number and unit
    # Supports: 100nM, 100 nM, 10µM, 10 uM, 0.5mM, etc.
    match = re.match(r'^(\d+\.?\d*)\s*([a-zA-Zµμ]+(?:/[a-zA-Z]+)?)?$', s)
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else None
            return (value, unit)
        except ValueError:
            pass

    return (None, None)


def _looks_like_slash_multivalue(value_str: str) -> bool:
    """True for LLM multi-cell dumps like '133/209/375', not for 'ng/mL'."""
    if not isinstance(value_str, str):
        return False
    if re.search(
        r"(?i)(ng|ug|µg|mg|pg|nmol|umol|µmol|mmol|pmol|fmol|mol)\s*/\s*(ml|l)\b",
        value_str,
    ):
        return False
    return bool(re.search(r"\d\s*/\s*\d", value_str))


def _parse_sandwich_range(s: str, default_unit: str | None = None) -> tuple | None:
    """
    Parse '20nM<ic50<10uM' / '100<ic50<500' / '20nm<ic50<10' (+ default_unit).

    Returns (avg, 'range', original_range) or None.
    When both ends are converted to nM, original_range ends with ' nM'.
    """
    # Optional units on either end; metric name in the middle (ic50, ki, ...)
    m = re.match(
        r'^(\d+\.?\d*(?:[eE][+\-]?\d+)?)\s*([a-zA-Zµμ]+)?'
        r'[<>=]+'
        r'[a-z0-9]+'
        r'[<>=]+'
        r'(\d+\.?\d*(?:[eE][+\-]?\d+)?)\s*([a-zA-Zµμ]+)?$',
        s,
    )
    if not m:
        return None
    try:
        v1 = float(m.group(1))
        u1 = m.group(2)
        v2 = float(m.group(3))
        u2 = m.group(4)
    except ValueError:
        return None

    if u1 is None and default_unit:
        u1 = default_unit
    if u2 is None and default_unit:
        u2 = default_unit

    if u1 is not None and u2 is not None:
        n1 = _convert_value_to_nM(v1, u1)
        n2 = _convert_value_to_nM(v2, u2)
        if n1 is not None and n2 is not None:
            lower, upper = min(n1, n2), max(n1, n2)
            avg = (lower + upper) / 2
            return (avg, "range", f"{lower} to {upper} nM")

    # Bare numbers (or unconvertible units): mean in raw units for process_row
    lower, upper = min(v1, v2), max(v1, v2)
    avg = (lower + upper) / 2
    return (avg, "range", f"{lower} to {upper}")


def _convert_value_to_nM(value: float, unit: str) -> float:
    """
    Convert a value with the given unit to nM.

    Args:
        value: numeric value
        unit: unit of measurement (nM, uM, µM, mM, pM, fM, M)

    Returns:
        Value in nM or None if conversion is impossible
    """
    if value is None or unit is None:
        return None

    unit_lower = unit.lower().replace('µ', 'u').replace('μ', 'u')

    # Conversion factors to nM
    conv = {
        'nm': 1,
        'um': 1000,
        'mm': 1_000_000,
        'pm': 0.001,
        'fm': 0.000001,
        'm': 1_000_000_000
    }

    if unit_lower in conv:
        return value * conv[unit_lower]

    return None


# Values above this (nM) for molar affinity metrics are treated as parse garbage.
IMPOSSIBLE_NM_LIMIT = 1_000_000.0  # 1 mM


# Step 2: Normalize value string (value)
def normalize_value(value_str, default_unit: str | None = None):
    """
    Convert a complexly formatted `value` string into a single numeric value (float).
    Improved version for handling complex cases.

    Args:
        value_str: raw value from the extractor
        default_unit: optional row unit applied to sandwich ends that lack a unit
            (e.g. ``20 nM<IC50<10`` + ``uM`` → right end is 10 µM)

    Returns:
        tuple: (value: float, relation: str, original_range: str)
        relation: "<", ">", "<=", ">=", "=" (default), "range" (for ranges)
        original_range: "X to Y" for ranges, "" otherwise.
            When ends were converted to nM, original_range ends with `` nM``.
    """
    if not isinstance(value_str, str):
        return (None, "=", "")

    if _looks_like_slash_multivalue(value_str):
        return (None, "=", "")

    s = value_str.lower().strip()

    # 1. Pre-clean text and special characters
    s = re.sub(
        r"\b(about|approx\.?|at least|or less|or greater|or higher|or more)\b", "", s
    )
    s = (
        s.replace(" ", "")
        .replace("x", "*")
        .replace("×", "*")
        .replace("−", "-")
    )
    # Handle ~ : between digits → range separator, at start → remove (means "approximately")
    s = re.sub(r'(\d)~(\d)', r'\1-\2', s)  # 50~500 → 50-500
    s = re.sub(r'^~', '', s)  # ~50 → 50
    s = s.replace("≦", "<=").replace("≧", ">=").replace("≤", "<=").replace("≥", ">=")
    s = s.replace(
        "*10-", "*10^-"
    )  # Standardize scientific notation for correct parsing

    # Handle "between X and Y"
    if "between" in s and "and" in s:
        parts = s.split("and")
        if len(parts) == 2:
            s = parts[0]  # Take the first, smaller value

    # 2. Handle "±" - take the value before the sign
    if "±" in s or "+-" in s or "-+" in s:
        s = re.split(r"±|\+-|-\+", s)[0]

    # strip trailing "+something" (like "7.9+0.8")
    m = re.match(r"^(\d*\.?\d+)\+\d*\.?\d+$", s)
    if m:
        return (float(m.group(1)), "=", "")

    # Extract leading relation after text cleanup, before sci-notation / single-value paths.
    # Sandwich/ranges that start with a digit keep "=" / "range".
    relation = "="
    rel_match = re.match(r'^([<>=]+)', s)
    if rel_match:
        relation = rel_match.group(1)
        s_rel = s[len(relation) :]
    else:
        s_rel = s

    # 3. Handle scientific notation (PRIORITY) — on the remainder after relation
    match = re.search(r"(\d*\.?\d*)\*?10\^\{?(-?\d+)\}?", s_rel)
    if match and re.fullmatch(
        r"[<>=]*\d*\.?\d*\*?10\^\{?-?\d+\}?[a-zA-Zµμ/]*", s
    ):
        try:
            base = float(match.group(1)) if match.group(1) else 1.0
            exponent = int(match.group(2))
            return (base * (10 ** exponent), relation, "")
        except OverflowError:
            return (None, "=", "")  # Number too large

    # Plain e-notation on a single value (possibly after stripping relation):
    # ">2.00e-5" → s_rel "2.00e-5". Reject strings that still look like ranges.
    if re.fullmatch(r"\d+\.?\d*[eE][+\-]?\d+[a-zA-Zµμ/]*", s_rel):
        num_part = re.match(r"(\d+\.?\d*[eE][+\-]?\d+)", s_rel)
        if num_part:
            try:
                return (float(num_part.group(1)), relation, "")
            except ValueError:
                pass

    # 4. Sandwich "number[unit][op]metric[op]number[unit]"
    # Examples: "20000>=IC50>=500", "100<IC50<500", "20nM<IC50<10uM", "20nm<ic50<10"
    sandwich = _parse_sandwich_range(s, default_unit=default_unit)
    if sandwich is not None:
        return sandwich

    # 5. Handle ranges "to", "-", "and" (AFTER scientific notation)
    # IMPORTANT: Now supports ranges with different units
    # Example: ">100 nM - 10 µM" → convert both parts to nM and compute the mean
    range_parts = []

    original_s = value_str.lower().strip().replace("−", "-")

    if " to " in original_s or (
        "to" in original_s and re.search(r"\dto\d|\d\s+to\s+\d|[eE][+\-]?\d+\s*to", original_s)
    ):
        range_parts = re.split(r"\s*to\s*", original_s, maxsplit=1)
    elif " - " in original_s:
        # Spaced separator - stronger range indicator
        range_parts = original_s.split(" - ")
    elif "-" in s:
        # Ensure hyphen is not part of a number (e.g. scientific notation)
        if not re.search(r"[eE]-", s):
            # Try pattern "number[unit]-number[unit]" (incl. sci notation)
            range_match = re.match(
                r'^[<>=]*(\d+\.?\d*(?:[eE][+\-]?\d+)?)\s*([a-zA-Zµμ]+)?'
                r'\s*-\s*'
                r'(\d+\.?\d*(?:[eE][+\-]?\d+)?)\s*([a-zA-Zµμ]+)?$',
                original_s,
            )
            if range_match:
                range_parts = [
                    f"{range_match.group(1)} {range_match.group(2) or ''}".strip(),
                    f"{range_match.group(3)} {range_match.group(4) or ''}".strip()
                ]
            else:
                range_parts = s.split("-")
    elif "and" in s and "between" not in value_str.lower():
        range_parts = re.split(r"\s+and\s+", original_s, maxsplit=1)

    if len(range_parts) == 2:
        # Try to extract values with units from both parts
        val1, unit1 = _extract_value_and_unit(range_parts[0])
        val2, unit2 = _extract_value_and_unit(range_parts[1])

        if val1 is not None and val2 is not None:
            # If both parts have units, convert to nM
            if unit1 is not None and unit2 is not None:
                val1_nM = _convert_value_to_nM(val1, unit1)
                val2_nM = _convert_value_to_nM(val2, unit2)

                if val1_nM is not None and val2_nM is not None:
                    lower = min(val1_nM, val2_nM)
                    upper = max(val1_nM, val2_nM)
                    avg = (lower + upper) / 2
                    original_range = f"{lower} to {upper} nM"
                    return (avg, "range", original_range)

            # If units match or are absent - legacy behavior
            lower = min(val1, val2)
            upper = max(val1, val2)
            avg = (lower + upper) / 2
            original_range = f"{lower} to {upper}"
            return (avg, "range", original_range)

        # Fallback: digit-strip ONLY when sci-notation was NOT present.
        # Otherwise "1.0E-08" → "1.008" (CLASS-1 digit glue).
        if any(re.search(r"[eE]", p) for p in range_parts):
            return (None, "=", "")
        try:
            lower = float(re.sub(r"[^0-9.]", "", range_parts[0]))
            upper = float(re.sub(r"[^0-9.]", "", range_parts[1]))
            avg = (lower + upper) / 2
            original_range = f"{lower} to {upper}"
            return (avg, "range", original_range)
        except (ValueError, IndexError):
            pass  # If that fails, try other methods

    # 6/7. Single-value path on remainder after leading relation
    s = re.sub(r"[<>=]", "", s_rel)
    s = s.replace(",", "")

    # Prefer float() on remaining sci / plain number before digit-stripping
    try:
        return (float(s), relation, "")
    except ValueError:
        pass

    # 8. Final direct conversion attempt
    # If the remainder still looks like unfinished sci-notation (digit + e + sign/digit),
    # do not digit-glue. Avoid matching the letter "e" inside words like "between".
    if re.search(r"\d[eE][+\-]?\d", s):
        return (None, "=", "")
    try:
        # Strip remaining non-numeric characters
        final_str = re.sub(r"[^0-9.]", "", s)
        if final_str:
            return (float(final_str), relation, "")
    except ValueError:
        return (None, "=", "")

    return (None, "=", "")


# Step 4: Normalize metric name (binding_metric)
def normalize_metric_name(metric_str):
    """
    Normalize metric name variants to one of the standard forms
    using regular expressions for flexibility.
    
    Supports:
    - Assay prefixes (cERK IC50, aERK2 IC50, MEK1/ERK2 IC50)
    - Logarithmic metrics (pIC50, pKi, pKd, pEC50, logIC50)
    - Various formats (IC-50, IC 50, IC50)
    - Dissociation constant = Kd
    
    IMPORTANT: This function does NOT strip p/log prefixes from the metric because they are used
    to detect logarithmic values in process_row() BEFORE calling this function.
    
    Examples:
        'IC50' -> ('IC50', False)
        'cERK IC50' -> ('IC50', False)
        'aERK2 IC50' -> ('IC50', False)
        'pIC50' -> ('IC50', True)
        'logIC50' -> ('IC50', True)
        'Ki' -> ('Ki', False)
        'pKi' -> ('Ki', True)
        'Dissociation Constant' -> ('Kd', False)
        'Inhibition' -> ('Inhibition', False)
    
    Returns:
        tuple: (metric_name, is_logarithmic) or (None, False) if unrecognized
    """
    if not isinstance(metric_str, str):
        return None, False

    s = metric_str.lower().strip().replace("₅₀", "50")
    is_logarithmic = False

    # Dict: {regex: standard name}
    # Patterns match the metric anywhere in the string, ignoring assay prefixes
    metric_patterns = {
        # IC50 - supports pIC50, logIC50, cERK IC50, IC-50, IC 50
        r"\b(p|log)\s*ic\s*[-]?\s*50\b": ("IC50", True),  # Logarithmic
        r"\bic\s*[-]?\s*50\b": ("IC50", False),  # Linear
        
        # Ki - supports pKi, logKi, pK_i, K_i
        r"\b(p|log)\s*k[_]?i\b": ("Ki", True),  # Logarithmic (pKi, pK_i)
        r"\bk[_]?i\b": ("Ki", False),  # Linear (Ki, K_i)
        
        # Kd - supports pKd, logKd, pK_D, K_D
        r"\b(p|log)\s*k[_]?d\b": ("Kd", True),  # Logarithmic (pKd, pK_D)
        r"\bk[_]?d\b": ("Kd", False),  # Linear (Kd, K_D)
        
        # Dissociation Constant = Kd
        r"dissociation\s+constant": ("Kd", False),
        
        # EC50 - supports pEC50, logEC50, EC-50, EC 50
        r"\b(p|log)\s*ec\s*[-]?\s*50\b": ("EC50", True),  # Logarithmic
        r"\bec\s*[-]?\s*50\b": ("EC50", False),  # Linear
        
        # Inhibition - percent inhibition
        r"^inhibition$": ("Inhibition", False),
        r"^%\s*inh": ("Inhibition", False),
        r"^%\s*inhibition": ("Inhibition", False),
        r"\binhibition\s*%": ("Inhibition", False),
    }

    # Check patterns in priority order (logarithmic first)
    for pattern, (name, is_log) in metric_patterns.items():
        if re.search(pattern, s):
            return name, is_log

    return None, False


# Step 5 (partial): Normalize unit name (unit) for linear metrics
def normalize_unit_name(unit_str):
    """
    Normalize unit name variants to a standard form
    using regular expressions.
    Supports both molar and mass concentrations.
    
    Returns:
        str: standard unit name (nM, uM, mM, pM, fM, M, ng/mL, µg/mL, mg/mL, pg/mL, %)
        tuple: (unit, multiplier) for /mL units (e.g. nmol/mL → ("nM", 1000))
        None: if the unit is unrecognized
    """
    if not isinstance(unit_str, str):
        return None

    s = unit_str.lower().strip()

    # Order matters! More specific patterns must come first.
    # /mL patterns return tuple (unit, multiplier) because 1 mL = 0.001 L
    unit_patterns = [
        # === Molar units with /mL (must be BEFORE /L patterns!) ===
        # Return tuple (unit, multiplier) - multiplier = 1000 because 1/mL = 1000/L
        (r"^nmol\s*/\s*ml$", ("nM", 1000)),
        (r"^[uμµ]mol\s*/\s*ml$", ("uM", 1000)),
        (r"^mmol\s*/\s*ml$", ("mM", 1000)),
        (r"^pmol\s*/\s*ml$", ("pM", 1000)),
        (r"^fmol\s*/\s*ml$", ("fM", 1000)),
        (r"^mol\s*/\s*ml$", ("M", 1000)),
        
        # === Alternative formats: "μmol/l", "μmol l^-1" (lowercase l, with optional ^-1) ===
        (r"^[uμµ]mol\s*/?\s*l(\^-1)?$", "uM"),
        (r"^nmol\s*/?\s*l(\^-1)?$", "nM"),
        (r"^mmol\s*/?\s*l(\^-1)?$", "mM"),
        (r"^pmol\s*/?\s*l(\^-1)?$", "pM"),
        (r"^fmol\s*/?\s*l(\^-1)?$", "fM"),
        (r"^mol\s*/?\s*l(\^-1)?$", "M"),
        
        # === Shorthand without /L (μmol = μM, nmol = nM, etc.) ===
        (r"^[uμµ]mol$", "uM"),
        (r"^nmol$", "nM"),
        (r"^pmol$", "pM"),
        (r"^fmol$", "fM"),
        (r"^mmol$", "mM"),
        
        # === microM, microMolar variations ===
        (r"^micro\s*m(olar)?$", "uM"),
        
        # === Standard molar units ===
        (r"^nano\s*molar$|^nm$", "nM"),
        (r"^micro\s*molar$|^[uμµ]m$", "uM"),
        (r"^milli\s*molar$|^mm$", "mM"),
        (r"^pico\s*molar$|^pm$", "pM"),
        (r"^femto\s*molar$|^fm$", "fM"),
        (r"^molar$|^m$", "M"),
        
        # === Mass concentrations (require molecular weight for conversion) ===
        (r"^ng/ml$|^ng\s*/\s*ml$", "ng/mL"),
        (r"^[uμµ]g/ml$|^[uμµ]g\s*/\s*ml$", "µg/mL"),
        (r"^mg/ml$|^mg\s*/\s*ml$", "mg/mL"),
        (r"^pg/ml$|^pg\s*/\s*ml$", "pg/mL"),
        
        # === Percentage units ===
        (r"^%$|^percent$", "%"),
    ]

    for pattern, result in unit_patterns:
        if re.search(pattern, s):
            return result

    return None

def _clean_unit(u: str) -> str:
    s = u.lower().strip()
    s = s.replace("µ", "u").replace("μ", "u")          # unify micro
    s = re.sub(r"\s+", "", s)                          # drop spaces
    s = s.replace("litre", "l").replace("liter", "l")  # unify liter
    s = s.replace("ca2+","")                           # drop ion notes
    s = s.replace("per","/")                           # unify per
    s = s.replace("・","/")                             # exotic divider
    return s

def is_activity_unit(u: str) -> bool:
    s = _clean_unit(u)
    return bool(re.search(r"/(min|sec|s|h|hr|day|mg|g|kg)", s))

# def normalize_unit_name(unit_str):
#     if not isinstance(unit_str, str):
#         return None
#     s = _clean_unit(unit_str)

#     # block activity/specific activity units
#     if is_activity_unit(s):
#         return None  # not convertible to nM

#     # direct molar units
#     if re.fullmatch(r"(n|u|m|p)?m", s):   # nM, uM, mM, pM, M
#         return {"nm":"nM","um":"uM","mm":"mM","pm":"pM","m":"M"}[s]

#     # concentration as amount/volume → normalize to nM
#     conc_map_to_nM = {
#         "nmol/l": 1,
#         "nmol/ml": 1000,
#         "umol/l": 1000,
#         "umol/ml": 1_000_000,
#         "mmol/l": 1_000_000,
#         "mmol/ml": 1_000_000_000,
#         "mol/l": 1_000_000_000,
#         "pmol/l": 0.001,
#         "pmol/ml": 1,
#     }
#     if s in conc_map_to_nM:
#         return ("nM", conc_map_to_nM[s])  # caller can use the multiplier

#     return None

def convert_mass_to_nM(value_mass, unit, molecular_weight):
    """
    Convert mass concentration (ng/mL, µg/mL, mg/mL, pg/mL) to molar (nM).
    
    Args:
        value_mass: concentration value in mass units
        unit: unit of measurement (ng/mL, µg/mL, mg/mL, pg/mL)
        molecular_weight: molecular weight in g/mol (Da)
    
    Returns:
        Concentration in nM or None if conversion is impossible
    
    Formula: nM = (mass_g/L) / (MW_g/mol) * 10^9
    Example: ng/mL = g/L * 10^-6
             nM = (ng/mL * 10^-6) / MW * 10^9 = (ng/mL * 1000) / MW
    """
    if not molecular_weight or molecular_weight <= 0:
        logging.warning(f"Invalid molecular weight: {molecular_weight}, cannot convert {unit} to nM")
        return None
    
    # Conversion factors to g/L
    mass_to_g_per_L = {
        "ng/mL": 1e-6,     # ng/mL → g/L: 1 ng/mL = 10^-6 g/L
        "µg/mL": 1e-3,     # µg/mL → g/L: 1 µg/mL = 10^-3 g/L
        "mg/mL": 1,        # mg/mL → g/L: 1 mg/mL = 1 g/L
        "pg/mL": 1e-9,     # pg/mL → g/L: 1 pg/mL = 10^-9 g/L
    }
    
    if unit not in mass_to_g_per_L:
        return None
    
    # Convert to g/L
    g_per_L = value_mass * mass_to_g_per_L[unit]
    
    # Convert to mol/L (M)
    mol_per_L = g_per_L / molecular_weight
    
    # Convert to nM (nmol/L)
    nM = mol_per_L * 1e9
    
    return round(nM, 3)


def convert_to_nM(value_float, original_unit, is_logarithmic=False, molecular_weight=None):
    """
    Convert a value to nM according to the unit type.
    
    Args:
        value_float: numeric value
        original_unit: original unit of measurement
        is_logarithmic: whether the value is logarithmic (pIC50, pKi, etc.)
        molecular_weight: molecular weight (required for mass units)
    
    Returns:
        Value in nM or None if conversion is impossible
    """
    if is_logarithmic:
        # pX → nM: pX = -log10[X(M)], X(M) = 10^(-pX)
        return (10 ** -value_float) * 1e9

    norm = normalize_unit_name(original_unit)
    if norm is None:
        return None  # unsupported or activity unit

    # Classic molar scales (including fM for femtomolar)
    conv = {"nM": 1, "uM": 1000, "mM": 1_000_000, "pM": 0.001, "fM": 0.000001, "M": 1_000_000_000}

    # Handle tuple (unit, multiplier) for /mL units
    # Example: nmol/mL → ("nM", 1000) means 1 nmol/mL = 1000 nM
    if isinstance(norm, tuple):
        unit, multiplier = norm
        if unit in conv:
            return round(value_float * conv[unit] * multiplier, 3)
        return None

    # Check whether the unit is a mass concentration
    mass_units = ["ng/mL", "µg/mL", "mg/mL", "pg/mL"]
    if norm in mass_units:
        return convert_mass_to_nM(value_float, norm, molecular_weight)
    
    # Check whether the unit is percent (for metrics like Inhibition)
    if norm == "%":
        # Percent values are not converted to nM; return as-is
        return value_float

    # Standard molar conversion
    if norm in conv:
        return round(value_float * conv[norm], 3)
    
    return None

def process_row(row):
    """
    Process one data row according to the normalization plan.
    Supports logarithmic values, mass concentrations, and percent metrics.
    """
    if not isinstance(row, dict):
        return None

    original_metric = row.get("binding_metric")
    original_value_str = row.get("value")
    original_unit = row.get("unit")
    if original_unit is None:
        original_unit = "nM"
        row["is_default_unit"] = False
    else:
        row["is_default_unit"] = True

    # Step 2: Normalize value and extract comparison operator
    # Pass row unit so sandwich ends without an explicit unit (``20 nM<IC50<10``)
    # can use it for the bare endpoint.
    value_float, relation, original_range = normalize_value(
        original_value_str, default_unit=original_unit
    )
    if value_float is None:
        return f"Wrong value `{original_value_str}`"

    # Step 3 and 4: Normalize metric name and detect logarithmic form
    # normalize_metric_name now returns (metric_name, is_logarithmic)
    if "adh" in original_metric or "adhesion" in original_metric:
        # The second token is the metric in labels like "cell adhesion IC50".
        # Single-word labels such as "Readhesion" have no second token, and
        # indexing blindly raised IndexError -- which the export reports as a
        # failure of the whole patent, not of the one unusable row. Fall back to
        # the label itself so normalize_metric_name rejects it as it should.
        adhesion_tokens = original_metric.split(" ")
        metric_source = adhesion_tokens[1] if len(adhesion_tokens) > 1 else original_metric
        metric_name, is_logarithmic = normalize_metric_name(metric_source)
        row["assay_context"] = "adhesion"
    else:
        metric_name, is_logarithmic = normalize_metric_name(original_metric)

    if metric_name is None:
        return f"Wrong metric `{original_metric}`"
    
    # Additional check: "log" in the value also indicates a logarithmic metric
    if isinstance(original_value_str, str) and "log" in original_value_str.lower():
        is_logarithmic = True

    # Step 5: Final processing and conversion
    # For mass units (ng/mL, µg/mL, mg/mL, pg/mL) do NOT convert here,
    # because molecular weight is needed and will be available later
    norm_unit = normalize_unit_name(original_unit)
    mass_units = ["ng/mL", "µg/mL", "mg/mL", "pg/mL"]
    # normalize_value already converted unit-bearing ranges to nM
    already_nM = isinstance(original_range, str) and original_range.endswith(" nM")
    
    if norm_unit in mass_units:
        # Keep value as-is; conversion happens later after MW is available
        final_value = value_float
        row["needs_mw_conversion"] = True  # Flag for later conversion
    elif metric_name == "Inhibition":
        # For percent metrics keep as-is
        final_value = value_float
        row["is_percentage"] = True  # Percent value flag
    elif already_nM and not is_logarithmic:
        final_value = value_float
        norm_unit = "nM"
    else:
        # For other metrics convert to nM
        final_value = convert_to_nM(value_float, original_unit, is_logarithmic)
        if final_value is None:
            return f"Cannot convert {original_unit} to nM for value {value_float}"

    # CLASS-1 guard: molar affinity metrics above 1 mM are almost always
    # digit-concatenation / unit-parse garbage from patent legends.
    molar_metrics = {"IC50", "EC50", "Ki", "Kd"}
    if (
        metric_name in molar_metrics
        and not row.get("needs_mw_conversion")
        and not row.get("is_percentage")
        and final_value is not None
        and final_value > IMPOSSIBLE_NM_LIMIT
    ):
        return (
            f"Impossible value {final_value} nM from `{original_value_str}` "
            f"(limit {IMPOSSIBLE_NM_LIMIT:g} nM)"
        )
    
    # Update the source row, preserving all other fields
    row["binding_metric"] = metric_name
    row["value"] = final_value
    row["relation"] = relation  # Comparison operator: <, >, <=, >=, =, range
    row["original_range"] = original_range  # For ranges: "X to Y"
    # CRITICAL: Preserve normalized unit for later conversion
    # If norm_unit is available (unit was recognized), use it
    # Otherwise keep original_unit (backward compatibility)
    row["unit"] = norm_unit if norm_unit else original_unit
    row["original_value"] = original_value_str
    row["original_unit"] = original_unit
    row["original_metric"] = original_metric
    row["is_logarithmic"] = is_logarithmic

    return row

def is_absolute(path):
    return os.path.isabs(path)

def read_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data, None
            else:
                return [], f"File {filepath} is not a list of dicts"
    except Exception as e:
        return [], f"Error reading {filepath}: {e}"

def collect_input_data(input_path, dir_path):
    """
    Collect all data from a file or from all *_resolved.json files in a directory.
    """
    errors = []
    all_data = []

    # Make path absolute
    if not is_absolute(input_path):
        input_path = os.path.join(dir_path, input_path)

    if os.path.isfile(input_path):
        data, err = read_json_file(input_path)
        all_data.extend(data)
        if err:
            errors.append(err)
    elif os.path.isdir(input_path):
        files = glob(os.path.join(input_path, "**", "*_resolved.json"), recursive=True)
        if not files:
            errors.append(f"No *_resolved.json files found in directory {input_path}")
        for file in files:
            data, err = read_json_file(file)
            name = Path(file).name
            patent_number = ""
            if isinstance(name, str):
                patent_number = name.split("_")[0]
            for row in data:
                row['patent_number'] = patent_number
                row['file_path'] = file
            all_data.extend(data)
            if err:
                errors.append(err)
    else:
        errors.append(f"Input path {input_path} is neither file nor directory")

    return all_data, errors

def main():
    patent_number = None
    dir_path = os.path.dirname(os.path.abspath(__file__))
    input_path = "./input/parquet_100_resolved"
    output_file = os.path.join(dir_path, "output/final_normalized_data.json")
    errors_file = os.path.join(dir_path, "output/skipped_rows.json")

    os.makedirs(os.path.join(dir_path, "output"), exist_ok=True)

    input_data, errors = collect_input_data(input_path, dir_path)

    normalized_data = []
    skipped_rows = []

    for row in input_data:
        processed = process_row(row)
        if isinstance(processed, dict):
            normalized_data.append(processed)
        else:
            row['reason'] = processed
            skipped_rows.append(row)

    print(f"Total rows processed: {len(input_data)}")
    print(f"Total rows after normalization and filtering: {len(normalized_data)}")
    print(f"Skipped rows: {len(skipped_rows)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    with open(errors_file, "w", encoding="utf-8") as f:
        json.dump(skipped_rows + [{"errors": errors}], f, indent=2, ensure_ascii=False)

    print(f"Normalized data saved to {output_file}")
    print(f"Skipped rows and errors saved to {errors_file}")

if __name__ == "__main__":
    main()