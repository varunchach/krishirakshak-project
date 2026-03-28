"""Format pipeline results into API responses."""

import json
from pathlib import Path


def load_disease_info(disease_name: str) -> dict:
    """Look up disease details from treatment knowledge base."""
    kb_path = Path("training/treatment_kb.json")
    with open(kb_path) as f:
        kb = json.load(f)["diseases"]
    return kb.get(disease_name, {})


def format_treatment(disease_name: str, kb_entry: dict) -> str:
    """Format treatment KB entry into readable text."""
    if not kb_entry:
        return f"Treatment information for {disease_name} is not available in our database."

    t = kb_entry.get("treatment", {})
    parts = [
        f"Immediate Action: {t.get('immediate_action', 'N/A')}",
        f"Recommended Pesticide: {t.get('pesticide', 'N/A')}",
        f"Application: {t.get('application', 'N/A')}",
        f"Prevention: {t.get('prevention', 'N/A')}",
    ]
    return " ".join(parts)
