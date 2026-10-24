# Capacity attribute correction 24 October 2026

The existing rule distinguishes reported data-center or campus scales from
identified supply quantities. It does not require future construction, measured
load or a utility contract before a project can enter a siting scenario.

| Source row | Record | Current decision | Evidence |
| --- | --- | --- | --- |
| 1152 | Joule | Exclude 6,227.5 MW midpoint from capacity weights | The interview identifies 455 MW of gas-engine delivery by 2026 and a 12-GW generation plan by 2032. These are different supply stages, not load-range bounds. |
| 1059 | GW Ranch | Exclude 1,000 MW from capacity weights | The developer identifies a private-grid generation campus and a 1-GW power-delivery stage, not a separately quantified data-center load. |
| 1664 | Ridgeline | Exclude 1,656 MW from capacity weights | The primary permit application, Attachment I, PDF page 32, reports turbine design capacity of 5,650 MMBtu/hr, approximately 1,656 MW thermal. It does not establish that electrical data-center demand. |
| 641 | New Era | Retain 7,000 MW project-scale weight | The developer separately advertises the Lea County campus as 7+ GW planned capacity. The weight is not calculated by summing gas and nuclear supply plans. |

Original source records are unchanged. All excluded sites remain in count-based
cases and the raw-attribute sensitivity; excluded MW cannot return through size
imputation. Exclusion of an attribute is not a claim of zero data-center demand.

The corrected default capture is 149,193 / 207,749 = 71.8140641%. The seven-region
capacity core has 267 records. The count capture remains 724 / 919. Both numerator
and national denominator are rebuilt before within-footprint normalization.

With the reference path, anticipated resources and published ordinary response,
MISO pressure is 10.0289598871 GW and its added requirement is 5.0391458871 GW.
The prospective path adds no requirement. The endpoint-only growth limit is
49.7540528%, versus 44.3348008% across all years. At the minimum pool-consistent
association, the matched ERCOT reference path requires 77.1712989% realization;
the background threshold remains 62.5213125%. The official-demand ERCOT threshold
remains 87.6443101%. These are specified scenario conditions.

National energy, official demand/resource/response ledgers, regional capacity
conventions, historical fits and PV dispatch inputs are unchanged. Only
allocation-dependent comparisons and reader exports are updated. Earlier versions
remain recoverable from Git history; the preceding public commit is
`6a16bbbe5b21ec1e4e36521021e85798bf01b6da`.

## Source locations

- Joule interview, 27 February 2026: https://about.bnef.com/insights/data-centers/us-data-center-builder-goes-off-grid-to-speed-deployments/
- GW Ranch developer description, accessed 24 October 2026: https://www.pacificoenergy.com/gw-ranch
- Ridgeline permit application, 18 March 2025, redacted public copy: https://westvirginiawatch.com/wp-content/uploads/2025/05/FD-air-permit-app.pdf
- New Era developer campus listing, accessed 24 October 2026: https://www.newerainfra.ai/
- New Era original announcement, 6 November 2025: https://investors.newerainfra.ai/news/news-details/2025/New-Era-Energy--Digital-Enters-into-Land-Option-Purchase-Agreement-for-3500-Acres-in-New-Mexico-for-7GW-AI-Data-Center-Hub-Marks-first-Wholly-Owned-Development-Separate-from-TCDC-Joint-Venture/default.aspx

## Rebuilding the extension inputs

After the normal baseline reproduction entry:

```bash
cd ../planning_conditions_20261023
python refresh_allocation_inputs.py --stage candidates
python run_pool_consistency.py
python refresh_allocation_inputs.py --stage conditions
python run_analysis.py
python run_paired_realization.py
python run_campus_retention.py
python -m unittest discover -p 'test_*.py'
```

The packaged extension inputs already contain this correction, so running the
baseline first is optional when using the extension by itself.
