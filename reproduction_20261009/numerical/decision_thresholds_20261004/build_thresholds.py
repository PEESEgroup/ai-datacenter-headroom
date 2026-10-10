#!/usr/bin/env python3
"""Reproduce conditional thresholds from frozen inputs without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
KEYS = ["planning_region", "year", "resource_condition"]
PAIR_KEYS = [*KEYS, "response_condition", "data_center_scenario_id", "post2030_extension_id",
             "footprint_capture_case", "peak_mapping_case", "allocation_case", "linked_response_fraction"]
TOL = 1e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_thresholds(panel):
    out = panel[panel.response_condition.eq("current_cap")].copy()
    k = 1 + out.reference_margin_level_pct / 100
    out["response_required_for_reference_GW"] = (out.gross_peak_demand_MW - out.capacity_equivalent_resources_MW / k) / 1000
    out["additional_net_peak_relief_GW"] = np.maximum(0, out.response_required_for_reference_GW - out.current_cap_response_MW / 1000)
    out["available_response_increase_GW"] = (out.published_response_MW - out.current_cap_response_MW) / 1000
    if (out.available_response_increase_GW < -TOL).any():
        raise ValueError("Published response must not be below the capped response")
    delta = out.available_response_increase_GW
    need = out.additional_net_peak_relief_GW
    out["required_fraction"] = np.where(need <= TOL, 0.0, np.divide(need, delta, out=np.full(len(out), np.nan), where=delta > TOL))
    out["threshold_status"] = np.select([need <= TOL, delta <= TOL, need > delta + TOL],
        ["above_reference_without_increase", "no_listed_increase_to_close_gap", "listed_increase_insufficient"],
        default="threshold_within_listed_increase")
    out["fraction_is_probability"] = False
    keep = [*KEYS, "resource_condition_type", "gross_peak_demand_MW", "capacity_equivalent_resources_MW",
            "reference_margin_level_pct", "current_cap_response_MW", "published_response_MW",
            "ordinary_response_MW", "provisional_large_load_response_MW", "response_required_for_reference_GW",
            "additional_net_peak_relief_GW", "available_response_increase_GW", "required_fraction", "threshold_status",
            "fraction_is_probability"]
    return out[keep].sort_values(KEYS).reset_index(drop=True)


def ercot_thresholds(response):
    out = response[response.planning_region.eq("ERCOT")].copy()
    need = np.maximum(0, out.response_required_for_reference_GW - out.ordinary_response_MW / 1000)
    available = out.provisional_large_load_response_MW / 1000
    out["required_provisional_curtailment_GW"] = need
    out["required_provisional_fraction"] = np.where(need <= TOL, 0.0, np.divide(need, available, out=np.full(len(out), np.nan), where=available > TOL))
    out["provisional_threshold_status"] = np.select([need <= TOL, available <= TOL, need > available + TOL],
        ["above_reference_without_provisional", "no_provisional_credit_to_close_gap", "provisional_credit_insufficient"],
        default="threshold_within_provisional_credit")
    return out.sort_values(KEYS).reset_index(drop=True)


def realization_paths(response, ercot, fractions):
    rows = []
    for basis, frame in [("increase_above_current_cap", response), ("ERCOT_provisional_only", ercot)]:
        for fraction in fractions:
            out = frame[[*KEYS, "resource_condition_type", "gross_peak_demand_MW", "capacity_equivalent_resources_MW", "reference_margin_level_pct"]].copy()
            if basis == "increase_above_current_cap":
                credited = frame.current_cap_response_MW + fraction * 1000 * frame.available_response_increase_GW
            else:
                credited = frame.ordinary_response_MW + fraction * frame.provisional_large_load_response_MW
            out["response_basis"] = basis
            out["realization_fraction"] = fraction
            out["credited_response_MW"] = credited.to_numpy()
            out["balance_GW"] = (out.capacity_equivalent_resources_MW - (1 + out.reference_margin_level_pct / 100) * (out.gross_peak_demand_MW - out.credited_response_MW)) / 1000
            out["required_contribution_GW"] = np.maximum(0, -out.balance_GW)
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def dc_thresholds(pairs):
    out = pairs.copy()
    load = out.candidate_gross_path_GW
    linked = out.candidate_linked_response_GW
    if (load < -TOL).any() or (linked < -TOL).any() or (linked > load + TOL).any():
        raise ValueError("Infeasible candidate response association")
    eta = np.divide(linked, load, out=np.zeros(len(out)), where=load > TOL)
    slope = (1 + out.reference_margin_level_pct / 100) * (1 - eta)
    b0 = out.source_path_removed_balance_GW
    out["associated_net_relief_rate"] = eta
    out["planning_balance_cost_per_GW_load"] = slope
    out["DC_load_at_reference_GW"] = np.divide(b0, slope, out=np.full(len(out), np.nan), where=(slope > TOL) & (b0 >= -TOL))
    out["DC_load_at_reference_GW"] = out.DC_load_at_reference_GW.clip(lower=0)
    out["DC_threshold_status"] = np.select([b0 < -TOL, slope <= TOL],
        ["background_already_below_reference", "no_finite_threshold_under_complete_association"],
        default="finite_conditional_load_threshold")
    out["threshold_minus_candidate_GW"] = out.DC_load_at_reference_GW - load
    out["balance_reconstruction_residual_GW"] = b0 - slope * load - out.replacement_balance_GW
    out["pressure_reconstruction_residual_GW"] = slope * load - out.dc_path_pressure_contribution_GW
    out["DC_added_requirement_check_GW"] = np.maximum(0, -out.replacement_balance_GW) - np.maximum(0, -b0)
    out["requirement_reconstruction_residual_GW"] = out.DC_added_requirement_check_GW - out.dc_added_required_contribution_GW
    out["threshold_is_physical_hosting_capacity"] = False
    return out.sort_values(PAIR_KEYS).reset_index(drop=True)


def display(frame, spec):
    mask = pd.Series(True, index=frame.index)
    for key, value in spec.items():
        if key.endswith("conditions") or key == "linked_response_fractions":
            continue
        mask &= frame[key].eq(value)
    mask &= frame.resource_condition.isin(spec["resource_conditions"])
    mask &= frame.response_condition.isin(spec["response_conditions"])
    mask &= frame.linked_response_fraction.isin(spec["linked_response_fractions"])
    return frame.loc[mask].copy()


def path_summary(paths):
    keys = ["planning_region", "resource_condition", "response_basis", "realization_fraction"]
    rows = []
    for key, group in paths.groupby(keys):
        group = group.sort_values("year")
        below = group.balance_GW < -TOL
        rows.append({**dict(zip(keys, key)), "years_below_reference": int(below.sum()),
            "first_below_reference_year": int(group.loc[below, "year"].min()) if below.any() else pd.NA,
            "maximum_requirement_GW": float(group.required_contribution_GW.max()),
            "summed_annual_requirement_GW_year": float(group.required_contribution_GW.sum()),
            "2035_balance_GW": float(group.loc[group.year.eq(2035), "balance_GW"].iloc[0]),
            "sum_is_unserved_energy": False})
    return pd.DataFrame(rows)


def timing_summary(panel):
    rows = []
    for (region, resource), group in panel[panel.response_condition.eq("published_all")].groupby(["planning_region", "resource_condition"]):
        ordered = group.sort_values("year")
        below = ordered.margin_GW < -TOL
        rows.append({"planning_region": region, "resource_condition": resource,
            "first_below_reference_year": int(ordered.loc[below, "year"].min()) if below.any() else pd.NA,
            "years_below_reference": int(below.sum()),
            "summed_annual_requirement_GW_year": float(np.maximum(0, -ordered.margin_GW).sum()),
            "maximum_requirement_GW": float(np.maximum(0, -ordered.margin_GW).max()),
            "2035_balance_GW": float(ordered.loc[ordered.year.eq(2035), "margin_GW"].iloc[0])})
    out = pd.DataFrame(rows)
    base = out[out.resource_condition.eq("anticipated")][["planning_region", "summed_annual_requirement_GW_year", "years_below_reference"]]
    out = out.merge(base.rename(columns={"summed_annual_requirement_GW_year": "source_summed_requirement_GW_year", "years_below_reference": "source_years_below_reference"}), on="planning_region", validate="many_to_one")
    out["additional_summed_requirement_GW_year"] = out.summed_annual_requirement_GW_year - out.source_summed_requirement_GW_year
    out["additional_years_below_reference"] = out.years_below_reference - out.source_years_below_reference
    return out


def fmt(value, signed=False):
    if pd.isna(value):
        return "--"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def fraction_label(value, status):
    if pd.isna(value):
        return "Unavailable"
    if value > 1 + TOL:
        return rf"{100 * value:.1f}\%$^{{\dagger}}$"
    return f"{100 * value:.1f}\\%"


def write_tables(response, ercot, selected, evidence, timing, target):
    target.mkdir(parents=True, exist_ok=True)
    ec = ercot[ercot.resource_condition.isin(["anticipated", "prospective"])]
    end = ec[ec.year.eq(2035)]
    lines = [r"\begin{table}[htbp]\centering", r"\caption{\textbf{Conditions that change the planning judgment.}}\label{tab:decision-thresholds}",
        r"\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{llrr}\toprule", r"ERCOT resources & Period & Needed relief (GW) & Provisional share \\\midrule"]
    for resource in ["anticipated", "prospective"]:
        frame = ec[ec.resource_condition.eq(resource)]
        row = frame.loc[frame.year.eq(2035)].iloc[0]
        horizon = frame.loc[frame.required_provisional_fraction.idxmax()]
        lines.append(f"{resource.capitalize()} & 2035 & {fmt(row.required_provisional_curtailment_GW)} & {fraction_label(row.required_provisional_fraction, '')} " + r"\\")
        lines.append(f" & All ten years & -- & {fraction_label(horizon.required_provisional_fraction, '')} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\par\vspace{0.8em}", r"\begin{tabular}{llrrr}\toprule", r"MISO resources & Association & No-new $B$ & Candidate $B$ & Load at $B=0$ \\\midrule"]
    mc = selected[selected.planning_region.eq("MISO") & selected.year.eq(2035) & selected.response_condition.eq("published_all")]
    for row in mc.sort_values(["resource_condition", "linked_response_fraction"]).itertuples():
        label = "Background below" if row.DC_threshold_status == "background_already_below_reference" else fmt(row.DC_load_at_reference_GW)
        lines.append(f"{row.resource_condition.capitalize()} & {100*row.linked_response_fraction:.0f}\\% & {fmt(row.source_path_removed_balance_GW, True)} & {fmt(row.replacement_balance_GW, True)} & {label} " + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\par\vspace{0.4em}\begin{minipage}{\textwidth}\footnotesize ERCOT holds annual ordinary response at its published value and varies the fraction of published provisional large-load curtailment. The all-ten-years value is the minimum constant fraction that meets every annual reference, not an expected realization. MISO uses the displayed LBNL Reference growth path (about 8.0 GW in 2035), published response and a fixed associated-response rate within each pair. MISO quantities are native GW; load thresholds are conditional peak-equivalent GW, not local hosting capacities. Association is a feasibility-bound assumption, not a measured DC-response fraction.\end{minipage}\end{table}"]
    lines.insert(-1, r"\par\vspace{0.3em}\noindent\footnotesize MISO association percentages refer to the feasible response bound, not to the fraction of DC load curtailed.\normalsize")
    (target / "main_thresholds.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\section{Supplementary Note 17: Conditions that change planning judgments}",
        "Thresholds invert the same frozen annual accounting, without fitting coefficients or changing the demand and resource paths. Response realization and DC response association are separate axes. A realization fraction is held constant over an entire ten-year path; the association fraction instead specifies which response is removed together with load in a paired comparison. Neither is a probability.",
        r"\subsection{Response needed under a stated resource path}",
        r"For $k=1+\mu$, the response required to meet reference is $F^{req}=G-R/k$. The additional net peak relief relative to current-capped response is $[F^{req}-F^{cap}]_+$. If the listed increase $D_F=F^{published}-F^{cap}$ is positive, the corresponding fraction is $\rho^*=[F^{req}-F^{cap}]_+/D_F$. A fraction above one means this listed increase is insufficient. If $D_F=0$ and relief is needed, the fraction is undefined rather than zero. Zero means the reference is already met, not that the system has no response.",
        r"\begin{longtable}{llrrl}\caption{Response thresholds in 2035 under source resource paths.}\label{tab:response-thresholds}\\",
        r"\toprule Region & Resources & Relief (GW) & Increase (GW) & Share \\\midrule\endfirsthead",
        r"\toprule Region & Resources & Relief (GW) & Increase (GW) & Share \\\midrule\endhead\bottomrule\endfoot"]
    for region in ["MISO", "PJM", "SPP", "ERCOT", "NYISO", "ISO-NE", "WECC-California"]:
        for row in response[response.planning_region.eq(region) & response.year.eq(2035) & response.resource_condition.isin(["anticipated", "prospective"])].sort_values("resource_condition").itertuples():
            lines.append(f"{'WECC-Calif.' if region == 'WECC-California' else region} & {row.resource_condition.capitalize()} & {fmt(row.additional_net_peak_relief_GW)} & {fmt(row.available_response_increase_GW)} & {fraction_label(row.required_fraction, row.threshold_status)} " + r"\\")
    lines += [r"\end{longtable}", r"\noindent Shares above 100\% ($\dagger$) cannot be met from the listed increase alone. Unavailable denotes a remaining gap with no listed increase. Native regional conventions are retained; these relief quantities are not assumed to be deliverable by data centers.",
        r"\subsection{ERCOT provisional large-load response}",
        r"Here $F(\zeta)=F^{ordinary}_t+\zeta F^{provisional}_t$. Annual ordinary response retains its source path, so $\zeta=0$ is not the current-cap comparator. The threshold is $\zeta^*=[G-R/k-F^{ordinary}]_+/F^{provisional}$. For one constant fraction over 2026--2035, the required fraction is the maximum annual threshold. The source reports a provisional curtailment credit and evolving emergency implementation, not demonstrated future performance. It includes large loads beyond data centers. The source's DC demand forecast already applies its reported 49.8\% observed-peak/request adjustment; this factor is not applied again\citep{nerc2025_ltra_revision}.",
        r"\begin{longtable}{lrrrr}\caption{Annual ERCOT provisional-response thresholds.}\label{tab:ercot-realization-thresholds}\\",
        r"\toprule Year & Published credit & A relief & A share & P share \\\midrule\endfirsthead",
        r"\toprule Year & Published credit & A relief & A share & P share \\\midrule\endhead\bottomrule\endfoot"]
    for year in range(2026, 2036):
        a = ec[ec.year.eq(year) & ec.resource_condition.eq("anticipated")].iloc[0]
        p = ec[ec.year.eq(year) & ec.resource_condition.eq("prospective")].iloc[0]
        lines.append(f"{year} & {fmt(a.provisional_large_load_response_MW/1000)} & {fmt(a.required_provisional_curtailment_GW)} & {fraction_label(a.required_provisional_fraction, '')} & {fraction_label(p.required_provisional_fraction, '')} " + r"\\")
    lines += [r"\end{longtable}", r"\noindent Credits and relief are GW. A/P denote anticipated/prospective resources. A zero threshold retains all ordinary response and the stated resource path; it does not imply that no adaptation is needed.",
        r"\subsection{Conditional DC growth threshold}",
        r"Within a matched background, let $B^0$ be the no-new-growth balance and $\eta=H^S/L^S$ the specified feasible associated-response rate. Then $B(L)=B^0-k(1-\eta)L$. If $B^0\geq0$ and $\eta<1$, $L^*=B^0/[k(1-\eta)]$. If $B^0<0$, the background already requires an additional contribution without the specified DC increment; no positive DC load can eliminate that gap. If $\eta=1$, this accounting boundary has no finite load threshold, which is not evidence of unlimited physical hosting capacity. The computation is performed within each scenario before ranges are summarized.",
        r"\begin{longtable}{llrrrr}\caption{Paired MISO balances and contributions for the 2035 threshold comparison.}\label{tab:dc-load-thresholds}\\",
        r"\toprule Resources & Association & $B^0$ & $B^S$ & $P^{DC,S}$ & $\Delta A^{DC,S}$ \\\midrule\endfirsthead",
        r"\toprule Resources & Association & $B^0$ & $B^S$ & $P^{DC,S}$ & $\Delta A^{DC,S}$ \\\midrule\endhead\bottomrule\endfoot"]
    for row in mc.sort_values(["resource_condition", "linked_response_fraction"]).itertuples():
        lines.append(f"{row.resource_condition.capitalize()} & {100*row.linked_response_fraction:.0f}\\% & {fmt(row.source_path_removed_balance_GW, True)} & {fmt(row.replacement_balance_GW, True)} & {fmt(row.dc_path_pressure_contribution_GW)} & {fmt(row.dc_added_required_contribution_GW)} " + r"\\")
    lines += [r"\end{longtable}", r"\noindent Quantities are native GW under published response. Load thresholds appear in main Table~\ref*{main-tab:decision-thresholds}. Complete annual thresholds retain the national, allocation, peak mapping and continuation identifiers rather than pooling unrelated backgrounds.",
        r"\subsection{Evidence and interpretation of the bounds}",
        r"\begin{longtable}{L{0.20\textwidth}L{0.28\textwidth}L{0.43\textwidth}}\caption{Evidence supporting the threshold inputs and remaining boundaries.}\label{tab:threshold-evidence}\\",
        r"\toprule Input & Evidence & Interpretation \\\midrule\endfirsthead",
        r"\toprule Input & Evidence & Interpretation \\\midrule\endhead\bottomrule\endfoot"]
    for row in evidence.itertuples():
        def clean(value):
            return str(value).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
        lines.append(" & ".join(clean(v) for v in [row.input, row.evidence, row.interpretation]) + r"\\")
    lines += [r"\end{longtable}",
        r"A sum of annual positive-part contribution requirements has units GW-year and describes the duration and scale of this planning-accounting condition; it is not unserved energy, outage duration or an expected loss. Likewise, a requirement under an expectation-based supply condition, $[Q-\mathrm{E}(R)]_+$, is not $\mathrm{E}[(Q-R)_+]$. No stochastic adequacy or operability inference is introduced by these thresholds.",
        "Under anticipated resources and published response, MISO's undelayed contribution path has six annual points below reference and a summed requirement of 69.76 GW-year. A two-year Tier 1 arrival delay gives ten points and 114.11 GW-year, an increase of 44.36 GW-year, while retaining the same 2035 balance. This additional timing stress is distinguished from the positive-net-resource-change delays in Note 6."]
    (target / "threshold_note.tex").write_text("\n".join(lines) + "\n")


def run(input_dir, output_dir, paper_dir=None):
    protocol = json.loads((HERE / "protocol.json").read_text())
    panel = pd.read_csv(input_dir / "native_panel.csv")
    pairs = pd.read_csv(input_dir / "paired_paths.csv.gz")
    evidence = pd.read_csv(input_dir / "source_evidence.csv")
    response = response_thresholds(panel)
    ercot = ercot_thresholds(response)
    paths = realization_paths(response, ercot, protocol["realization_fractions"])
    dc = dc_thresholds(pairs)
    selected = display(dc, protocol["display_selection"])
    timing = timing_summary(panel)
    summaries = path_summary(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"01_response_thresholds.csv": response, "02_ercot_provisional_thresholds.csv": ercot,
        "03_response_realization_annual_paths.csv": paths, "04_response_realization_summaries.csv": summaries,
        "05_DC_load_thresholds.csv": dc, "06_displayed_DC_thresholds.csv": selected,
        "07_timing_duration_and_requirement.csv": timing}
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, float_format="%.12g", na_rep="")
    checks = []
    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    check("unique_response_keys", not response.duplicated(KEYS).any())
    check("350_annual_resource_thresholds", len(response) == 350)
    check("complete_response_years", response.groupby(["planning_region", "resource_condition"]).year.nunique().eq(10).all())
    identifiable = response[response.required_fraction.gt(TOL) & response.required_fraction.notna()]
    closure = (identifiable.current_cap_response_MW / 1000 + identifiable.required_fraction * identifiable.available_response_increase_GW - identifiable.response_required_for_reference_GW).abs()
    check("inverse_response_threshold_closes_reference", closure.max() < TOL)
    check("unique_pair_keys", not dc.duplicated(PAIR_KEYS).any())
    check("388800_existing_pairs_retained", len(dc) == 388800)
    check("complete_paired_annual_paths", dc.groupby([k for k in PAIR_KEYS if k != "year"]).year.nunique().eq(10).all())
    check("240_displayed_pairs_retained", len(selected) == 240)
    for name in ["balance", "pressure", "requirement"]:
        check(name + "_identity", dc[name + "_reconstruction_residual_GW"].abs().max() < TOL)
    finite = dc[dc.DC_threshold_status.eq("finite_conditional_load_threshold")]
    check("finite_DC_threshold_closes_balance", (finite.source_path_removed_balance_GW - finite.planning_balance_cost_per_GW_load * finite.DC_load_at_reference_GW).abs().max() < TOL)
    below = dc[dc.DC_threshold_status.eq("background_already_below_reference")]
    check("negative_background_not_zero_capacity", below.DC_load_at_reference_GW.isna().all())
    rp = paths[paths.response_basis.eq("increase_above_current_cap")]
    for fraction, condition in [(0, "current_cap"), (1, "published_all")]:
        exact = rp[rp.realization_fraction.eq(fraction)].merge(panel[panel.response_condition.eq(condition)][[*KEYS, "margin_GW"]], on=KEYS, validate="one_to_one")
        check("generic_fraction_" + str(fraction) + "_reproduces_" + condition, (exact.balance_GW - exact.margin_GW).abs().max() < TOL)
    ep = paths[paths.response_basis.eq("ERCOT_provisional_only") & paths.realization_fraction.eq(1)]
    exact = ep.merge(panel[panel.planning_region.eq("ERCOT") & panel.response_condition.eq("published_all")][[*KEYS, "margin_GW"]], on=KEYS, validate="one_to_one")
    check("ERCOT_full_provisional_reproduces_published", (exact.balance_GW - exact.margin_GW).abs().max() < TOL)
    ordered = paths.sort_values([*KEYS, "response_basis", "realization_fraction"])
    check("response_monotonicity", ordered.groupby([*KEYS, "response_basis"]).balance_GW.diff().dropna().ge(-TOL).all())
    check("no_probability_claim", not response.fraction_is_probability.any())
    check("no_unserved_energy_claim", not summaries.sum_is_unserved_energy.any())
    qa = pd.DataFrame(checks)
    qa.to_csv(output_dir / "08_QA.csv", index=False)
    facts = {"QA_passed": bool(qa.passed.all()), "QA_count": len(qa), "input_rows": {"native_panel": len(panel), "paired_paths": len(pairs)},
        "ERCOT_source_horizon_thresholds": {}, "MISO_2035_thresholds": [], "MISO_two_year_timing": {}}
    for resource in protocol["source_resources"]:
        frame = ercot[ercot.resource_condition.eq(resource)]
        worst = frame.loc[frame.required_provisional_fraction.idxmax()]
        end = frame.loc[frame.year.eq(2035)].iloc[0]
        facts["ERCOT_source_horizon_thresholds"][resource] = {"2035_fraction": float(end.required_provisional_fraction), "2035_relief_GW": float(end.required_provisional_curtailment_GW), "horizon_fraction": float(worst.required_provisional_fraction), "binding_year": int(worst.year)}
    for row in selected[selected.planning_region.eq("MISO") & selected.year.eq(2035) & selected.response_condition.eq("published_all")].itertuples():
        facts["MISO_2035_thresholds"].append({"resources": row.resource_condition, "association": row.linked_response_fraction,
            "load_GW": row.candidate_gross_path_GW, "threshold_GW": None if pd.isna(row.DC_load_at_reference_GW) else row.DC_load_at_reference_GW,
            "status": row.DC_threshold_status, "no_new_balance_GW": row.source_path_removed_balance_GW, "candidate_balance_GW": row.replacement_balance_GW})
    mt = timing[timing.planning_region.eq("MISO") & timing.resource_condition.eq("anticipated_tier1_arrival_delayed_2y")].iloc[0]
    facts["MISO_two_year_timing"] = mt.to_dict()
    (output_dir / "facts.json").write_text(json.dumps(facts, indent=2))
    manifest = [{"file": name, "sha256": digest(input_dir / name)} for name in ["native_panel.csv", "paired_paths.csv.gz", "source_evidence.csv"]]
    manifest += [{"file": name, "sha256": digest(output_dir / name)} for name in outputs]
    pd.DataFrame(manifest).to_csv(output_dir / "09_manifest.csv", index=False)
    if paper_dir is not None:
        write_tables(response, ercot, selected, evidence, timing, paper_dir / "tables/decision_thresholds_20261004")
    print(json.dumps(facts, indent=2))
    if not qa.passed.all():
        raise AssertionError("Threshold QA did not pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "inputs")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.paper_dir)
