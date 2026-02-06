import pandas as pd
import numpy as np

YES = {"yes", "y", "true", "t", "1", 1, True}
NO  = {"no", "n", "false", "f", "0", 0, False}

# Map messy source column names -> standardized (snake-ish) names
RENAME_MAP = {
    "Threat Type": "threat_type",
    "Weather_Severity": "weather_severity",
    "Intel Confidence": "intel_confidence",
    "force_readiness_score": "readiness_level",
    "roe_complexity_score": "roe_complexity",
    "Operational Budget (MUSD)": "budget_musd",
    "Theater Distance KM": "distance_to_theater_km",
    "logistics_delay_hours": "logistics_delay_hours",
    "friendlyUnitCount": "friendly_unit_count",
    "ISR_AssetCount": "isr_asset_count",
    "Patriot.Batteries": "patriot_batteries",
    "Enemy.Capability.Index": "enemy_capability_index",
    "Aircraft Count": "aircraft_count",
    "Supply Chain Resilience": "supply_chain_resilience",
    "BudgetUtilization_pct": "budget_utilization_pct",
    "CMD_COORD_SCORE": "cmd_coord_score",
    "JointForceIntegration": "joint_force_integration",
    "satellite coverage score": "satellite_coverage_score",
    "LCS_COUNT": "lcs_count",
    "ResponseTime_hrs": "response_time_hrs",
    "ThreatEscalationHours": "threat_escalation_hours",
    "PriorEngagements": "prior_engagements",
    # keep Season as "Season" in source; we normalize it but keep name "Season" unless you prefer "season"
}

def _to_bool01(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in YES:
            return 1
        if s in NO:
            return 0
        if s in {"success", "succeeded"}:
            return 1
        if s in {"fail", "failed"}:
            return 0
        return np.nan
    if x in YES:
        return 1
    if x in NO:
        return 0
    return np.nan

def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize object/string columns:
    - cast to pandas 'string' dtype (keeps NA as <NA>)
    - strip whitespace
    - convert common null strings to NA
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object" or pd.api.types.is_string_dtype(out[c]):
            s = out[c].astype("string").str.strip()
            s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA, "null": pd.NA, "NULL": pd.NA})
            out[c] = s
    return out

def normalize_season(s: pd.Series) -> pd.Series:
    """
    Collapse season variants:
    - strips, lowercases
    - maps autumn -> fall
    - invalid values -> NA
    """
    s = s.astype("string").str.strip().str.lower()
    s = s.replace({"autumn": "fall"})
    valid = {"spring", "summer", "fall", "winter"}

    # mask invalid values -> <NA> for pandas StringDtype
    s = s.mask(~s.isin(valid))

    return s


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def clean_prompt2(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns (clean_df, report_dict).

    What this does:
    1) Normalizes string columns
    2) Renames messy source columns into a consistent schema
    3) Normalizes response_success into 0/1
    4) Normalizes Season into {spring, summer, fall, winter}
    5) Coerces key numeric columns
    6) Adds bad_success flag
    """
    report: dict = {}
    out = normalize_strings(df)

    # Rename columns into standardized schema
    before_cols = list(out.columns)
    out = out.rename(columns=RENAME_MAP)
    after_cols = list(out.columns)
    report["renamed_columns"] = {k: v for k, v in RENAME_MAP.items() if k in before_cols}
    report["columns_before"] = before_cols
    report["columns_after"] = after_cols

    # Normalize Season values (keep column name "Season" for now)
    if "Season" in out.columns:
        out["Season"] = normalize_season(out["Season"])

    # Normalize threat_type casing if present
    if "threat_type" in out.columns:
        out["threat_type"] = out["threat_type"].astype("string").str.strip().str.lower()

    # --- target normalization ---
    if "response_success" in out.columns:
        before = out["response_success"].value_counts(dropna=False).to_dict()
        out["response_success"] = out["response_success"].apply(_to_bool01)
        after = out["response_success"].value_counts(dropna=False).to_dict()
        report["response_success_value_counts_before"] = before
        report["response_success_value_counts_after"] = after

    # Coerce numerics (use standardized names)
    numeric_candidates = [
        # outcomes
        "Financial_Loss_MUSD",
        "actual_days_to_stabilization",
        # core features
        "enemy_unit_count",
        "friendly_unit_count",
        "logistics_delay_hours",
        "distance_to_theater_km",
        "roe_complexity",
        "weather_severity",
        "readiness_level",
        "intel_confidence",
        "budget_musd",
        # other useful features that appear in your dataset
        "aircraft_count",
        "isr_asset_count",
        "patriot_batteries",
        "enemy_capability_index",
        "supply_chain_resilience",
        "budget_utilization_pct",
        "cmd_coord_score",
        "joint_force_integration",
        "satellite_coverage_score",
        "lcs_count",
        "response_time_hrs",
        "threat_escalation_hours",
        "prior_engagements",
        "cyber_defense_teams",
    ]
    out = coerce_numeric(out, numeric_candidates)

    # Create a “bad outcome” flag for quick EDA (for success)
    if "response_success" in out.columns:
        out["bad_success"] = (out["response_success"] == 0).astype("Int64")
    # Convert nullable string columns back to object dtype for sklearn compatibility
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].astype("object")
    # Missingness snapshot
    report["missingness_pct"] = (out.isna().mean().sort_values(ascending=False) * 100).round(2).to_dict()

    return out, report
