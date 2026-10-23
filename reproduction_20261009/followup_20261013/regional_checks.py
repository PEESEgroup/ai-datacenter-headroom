"""Reproduce the native-ledger comparisons cited in the reviewer response."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(HERE / "expected/regional_input_reconciliation.csv")
    np.testing.assert_allclose(rows.resource_GW - rows.requirement_GW, rows.balance_GW, atol=1e-10)
    normalized = rows.copy()
    normalized["native_balance_as_pct_requirement"] = 100 * rows.balance_GW / rows.requirement_GW
    normalized["capacity_convention"] = rows.apply(
        lambda row: "NERC native contribution / reference requirement" if row["case"].startswith("NERC")
        else ("Summer SAC / PRMR" if row.region == "MISO" else row["case"].removeprefix("RRS-")), axis=1)
    normalized["interpretation"] = "Own-study normalized ledger; not equal LOLE, common-portfolio accreditation or forecast error"
    normalized.to_csv(output / "regional_native_comparison.csv", index=False)

    identities = []
    for row in rows[rows["case"].str.startswith("NERC")].itertuples():
        factor = row.requirement_GW / (row.gross_GW - row.separate_response_GW)
        requirement = factor * row.gross_GW
        response_credit = factor * row.separate_response_GW
        contribution = row.resource_GW + response_credit
        balance = contribution - requirement
        np.testing.assert_allclose(balance, row.balance_GW, atol=1e-10)
        identities.append(dict(
            region=row.region, summer_year=row.summer_year, case=row.case,
            native_resource_GW=row.resource_GW, gross_peak_GW=row.gross_GW,
            separate_response_GW=row.separate_response_GW, native_requirement_factor=factor,
            gross_requirement_GW=requirement, response_accounting_credit_GW=response_credit,
            equivalent_gross_side_contribution_GW=contribution, preserved_balance_GW=balance,
            interpretation="Algebraic restatement only; multiplied response credit is not newly accredited physical capacity"))
    pd.DataFrame(identities).to_csv(output / "nerc_response_side_identity.csv", index=False)

    mapping = []
    factors = pd.read_csv(HERE / "inputs/pjm_rrs_table4.csv")
    for row in factors.itertuples():
        part = rows[rows.region.eq("PJM") & rows.summer_year.eq(row.summer_year)].set_index("case")
        icap, ucap = part.loc["RRS-ICAP"], part.loc["RRS-UCAP"]
        factor = row.pool_accreditation_pct / 100
        irm = row.irm_pct / 100
        np.testing.assert_allclose(icap.gross_GW, ucap.gross_GW)
        for field in ["resource_GW", "requirement_GW", "balance_GW"]:
            assert abs(icap[field] * factor - ucap[field]) < .025
        assert abs((1 + irm) * factor - row.fpr) < .0001
        balance = icap.balance_GW * factor
        mapping.append(dict(
            summer_year=row.summer_year, own_RRS_pool_AUCAP_factor=factor,
            own_RRS_IRM=irm, own_RRS_FPR=row.fpr,
            RRS_ICAP_balance_GW=icap.balance_GW, mapped_RRS_UCAP_balance_GW=balance,
            reported_RRS_UCAP_balance_GW=ucap.balance_GW,
            mapping_residual_MW=1000 * (ucap.balance_GW - balance), applied_to_NERC=False,
            interpretation="RRS's own portfolio only; rounding/model reporting residual retained"))
    pd.DataFrame(mapping).to_csv(output / "pjm_within_report_mapping.csv", index=False)

    names = ["regional_native_comparison.csv", "nerc_response_side_identity.csv", "pjm_within_report_mapping.csv"]
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(output / name), pd.read_csv(HERE / "expected" / name),
                                      check_dtype=False, rtol=1e-10, atol=1e-10)
    report = dict(passed=True, native_rows=len(rows), response_identity_rows=len(identities),
                  own_PJM_mapping_rows=len(mapping), cross_study_accreditation_conversion=False)
    (output / "regional_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    print(json.dumps(build(parser.parse_args().output_root), indent=2))
