"""Utilities for state name canonicalization and validation."""

from typing import Dict, Set


def get_state_alias_map() -> Dict[str, str]:
    """
    Return a minimal alias->canonical mapping for Indian states/UTs.
    Keys and values are case-insensitive; caller should lowercase/strip before lookup.
    Extend this list as needed.
    """
    aliases = {
        # Common abbreviations
        "up": "Uttar Pradesh",
        "wb": "West Bengal",
        "uk": "Uttarakhand",
        "mp": "Madhya Pradesh",
        "hp": "Himachal Pradesh",
        "jk": "Jammu and Kashmir",
        "j&k": "Jammu and Kashmir",
        "tn": "Tamil Nadu",
        "ap": "Andhra Pradesh",
        "ts": "Telangana",
        "mh": "Maharashtra",
        "gj": "Gujarat",
        "pb": "Punjab",
        "rj": "Rajasthan",
        "br": "Bihar",
        "kl": "Kerala",
        "ka": "Karnataka",
        "cg": "Chhattisgarh",
        "ct": "Chhattisgarh",
        "od": "Odisha",
        "or": "Odisha",
        "nl": "Nagaland",
        "as": "Assam",
        "arpr": "Arunachal Pradesh",
        "ar": "Arunachal Pradesh",
        "sk": "Sikkim",
        "tr": "Tripura",
        "mz": "Mizoram",
        "ml": "Meghalaya",
        "ga": "Goa",
        "dl": "Delhi",
        "ch": "Chandigarh",
        "ld": "Lakshadweep",
        "py": "Puducherry",
        "an": "Andaman and Nicobar Islands",
        "dn": "Dadra and Nagar Haveli and Daman and Diu",
        "dd": "Dadra and Nagar Haveli and Daman and Diu",
        "jk&l": "Ladakh",  # old style
        # Historical/variant names
        "orissa": "Odisha",
        "uttaranchal": "Uttarakhand",
        "pondicherry": "Puducherry",
        "jammu & kashmir": "Jammu and Kashmir",
        "jammu and kashmir": "Jammu and Kashmir",
    }
    # normalize keys to lowercase
    aliases = {k.lower().strip(): v for k, v in aliases.items()}
    return aliases


def canonicalize_state_name(name: str, canonical_set: Set[str]) -> str:
    """
    Map input name/alias to canonical form if possible, else return original.
    """
    if not name:
        return ""
    s = str(name).strip()
    alias_map = get_state_alias_map()
    # direct match
    if s in canonical_set:
        return s
    # alias match
    lower = s.lower()
    if lower in alias_map:
        return alias_map[lower]
    return s


def is_valid_state_name(name: str, canonical_set: Set[str]) -> bool:
    """
    Validate that a state string matches (or aliases to) a canonical state name.
    """
    if not name:
        return False
    return canonicalize_state_name(name, canonical_set) in canonical_set


