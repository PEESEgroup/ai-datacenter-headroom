"""Evidence flags and rule-specific campus sensitivities, not realized-use labels."""
import re

import numpy as np
import pandas as pd

EVIDENCE_FIELDS = ["purpose", "other_info", "facility_name", "information_source"] + [f"info_source_{i}" for i in range(1,9)]
CRYPTO = r"\b(?:cryptocurrency|crypto(?:[ -]?min(?:e|es|ing))?|bitcoin|mining)\b"
OTHER_USE = r"\b(?:AI|IA|HPC|colocation)\b|artificial intelligence|high.performance"
TRANSITION = r"transition|convert|conversion|pivot|demolish|potential|former"
PLACEHOLDER_ADDRESS = r"\b(?:unknown|undisclosed|unspecified|unannounced|tbd|pending)\b"
DISTANCE_ADDRESS = r"^\d+(?:[ .]\d+)?\s+(?:miles?|mi|kilometers?|kilometres?|km|feet|ft|meters?|metres?)\b(?:\s+(?:outside|from|north|south|east|west|away|beyond|off|near|of|to)\b|$)"


def classify_scope(rows):
    out = rows.copy()
    texts = {name:out.get(name,pd.Series("",index=out.index)).fillna("").astype(str) for name in EVIDENCE_FIELDS}
    combined = pd.Series([" | ".join(texts[name].loc[i] for name in EVIDENCE_FIELDS) for i in out.index],index=out.index)
    narrative = texts["purpose"] + " | " + texts["other_info"]
    crypto = combined.str.contains(CRYPTO,case=False,regex=True).fillna(False).to_numpy(dtype=bool)
    other = narrative.str.contains(OTHER_USE,case=False,regex=True).fillna(False).to_numpy(dtype=bool)
    transition = narrative.str.contains(TRANSITION,case=False,regex=True).fillna(False).to_numpy(dtype=bool)
    purpose_crypto = texts["purpose"].str.contains(CRYPTO,case=False,regex=True).fillna(False).to_numpy(dtype=bool)
    mixed = crypto & (other | transition)
    pure = crypto & purpose_crypto & ~mixed
    out["scope_class"] = np.select([mixed,pure,crypto],
        ["mixed_or_transition_crypto","explicit_pure_crypto","uncertain_crypto_reference"],default="no_explicit_crypto_tag")
    out["crypto_evidence_fields"] = [";".join(name for name in EVIDENCE_FIELDS if re.search(CRYPTO,texts[name].loc[i],re.I)) for i in out.index]
    out["crypto_evidence_text"] = combined.where(crypto,"")
    out["current_purpose_raw"] = texts["purpose"]
    out["transition_or_uncertain_evidence"] = crypto & transition
    out["crypto_MW_fraction_identified"] = False
    return out


def campus_keys(rows,normalize):
    address = rows.address.apply(normalize)
    placeholders = address.str.contains(PLACEHOLDER_ADDRESS,regex=True).fillna(False)
    # A route number or road vicinity is not a premise identifier.
    street_like = address.str.match(r"^\d+[a-z]?\s+\S+") & address.str.len().ge(8)
    distance_description = address.str.contains(DISTANCE_ADDRESS, regex=True).fillna(False)
    usable = street_like & ~placeholders & ~distance_description
    keys = "row|" + rows.source_row.astype(str)
    county = rows.get("county_norm", rows.get("county", pd.Series("", index=rows.index))).apply(normalize)
    # Identical street strings in different counties do not establish one campus.
    keys.loc[usable] = rows.loc[usable,"state"].fillna("").astype(str) + "|" + county.loc[usable] + "|" + address.loc[usable]
    return keys, usable.to_numpy(dtype=bool)


def campus_allocation_cases(rows,rules):
    _, subsets = rules.allocation_cases(rows)
    outputs, selections = [], []
    for case, eligible in subsets.items():
        # Eligibility, mapping and positive weight are resolved before campus selection.
        chosen = eligible.sort_values(["reported_mw_mid","source_row"],ascending=[False,True],na_position="last").drop_duplicates("campus_key")
        total = chosen.allocation_weight.sum()
        assert total > 0
        grouped = chosen.groupby("ISO_ASSIGNED").allocation_weight.sum()
        for region in rules.REGIONS:
            value = float(grouped.get(region,0.0))
            outputs.append({"allocation_case":case,"model_region":region,
                "records_used":int(chosen.ISO_ASSIGNED.eq(region).sum()),"allocation_weight":value,
                "allocation_share":value/total,"allocated_per_100_GW":100*value/total,
                "fixed_total_rule":"renormalized_to_same_scenario_year_total",
                "weight_interpretation":"locational evidence, not realized load"})
        selected = chosen[["source_row","facility_name","status","campus_key","allocation_weight"]].copy()
        selected["allocation_case"] = case
        selections.append(selected)
    return pd.DataFrame(outputs),pd.concat(selections,ignore_index=True)
