"""Portable accounting primitive used by the unchanged rolling-origin refit."""
import numpy as np
import pandas as pd


def account(demand, response, resources, rml):
    out = pd.DataFrame({"gross_peak_demand_MW": np.atleast_1d(demand).astype(float),
                        "credited_DR_MW": np.atleast_1d(response).astype(float),
                        "capacity_equivalent_resources_MW": np.atleast_1d(resources).astype(float),
                        "reference_margin_level_pct": np.atleast_1d(rml).astype(float)})
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Nonfinite accounting inputs")
    out["net_peak_demand_MW"] = out.gross_peak_demand_MW - out.credited_DR_MW
    if out.net_peak_demand_MW.le(0).any():
        raise ValueError("Nonpositive net demand")
    out["reference_requirement_MW"] = (1 + out.reference_margin_level_pct / 100) * out.net_peak_demand_MW
    out["margin_MW"] = out.capacity_equivalent_resources_MW - out.reference_requirement_MW
    out["gap_to_RML_pp"] = 100 * out.margin_MW / out.net_peak_demand_MW
    return out
