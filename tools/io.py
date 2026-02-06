from pathlib import Path
import json
import pandas as pd

def load_prompt2_json(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw could be list[dict] or dict with a key holding rows
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        # common patterns
        for key in ["data", "rows", "records", "items"]:
            if key in raw and isinstance(raw[key], list):
                rows = raw[key]
                break
        else:
            # fallback: if dict itself is a row
            rows = [raw]
    else:
        raise ValueError(f"Unexpected JSON format: {type(raw)}")

    return pd.DataFrame(rows)
