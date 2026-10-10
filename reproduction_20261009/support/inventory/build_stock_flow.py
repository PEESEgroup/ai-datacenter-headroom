"""Physical stock-flow audit and fixed-origin contribution-index experiment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / 'inputs'
from pypdf import PdfReader

spec = importlib.util.spec_from_file_location('eia_parser', HERE / 'eia_parser.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
SHEETS = ['Operating', 'Planned', 'Retired', 'Canceled or Postponed']
MODELS = ['inventory_persistence', 'announced_schedule', 'learned_visible_stock', 'learned_open_stock']
KEYS = ['vintage', 'target_year', 'region']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(year):
    return INPUTS / f'eia860m/june_generator{year}.xlsx'


def technology(row):
    fuel = str(row.get('Energy Source Code', '')).strip().upper()
    prime = str(row.get('Prime Mover Code', '')).strip().upper()
    text = str(row.get('Technology', '')).lower()
    if (fuel == 'MWH' or 'battery' in text or 'energy storage' in text
            or 'pumped storage' in text or prime in ['BA', 'CE', 'FW']):
        return 'storage'
    if fuel == 'WND':
        return 'wind'
    if fuel == 'SUN':
        return 'solar'
    if fuel == 'NUC':
        return 'nuclear'
    if fuel == 'WAT':
        return 'hydro'
    if fuel in ['BIT', 'SUB', 'LIG', 'WC', 'RC', 'ANT', 'SGC']:
        return 'coal'
    if fuel in ['NG', 'BFG', 'OG', 'PG', 'SG']:
        return 'gas'
    return 'other'


def numeric(frame, name):
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors='coerce')


def due(year, month, target):
    """Missing month only resolves when the year is strictly before target."""
    return ((year < target) | ((year == target) & (month <= 6))).fillna(False)


def identify_shared_capacity(ledger):
    """Keep EIA reporting carriers; never invent engineering ratings for blank members."""
    ledger = ledger.copy()
    ledger['shared_capacity_member'] = False
    ledger['aggregate_capacity_carrier'] = False
    ledger['shared_capacity_ambiguous'] = False
    audit = []
    candidates = ledger[ledger.unit_code.ne('')]
    for (year, plant, code), group in candidates.groupby(['snapshot_year', 'plant_id', 'unit_code']):
        missing = group.summer_MW.isna() | group.summer_MW.lt(0)
        reported = group.summer_MW.gt(0)
        if not missing.any() or not reported.any() or len(group) < 2:
            continue
        carrier = group[reported]
        consistent = (group.sheet.nunique() == 1 and group.BA.nunique() == 1
            and group.status.nunique() == 1 and group.technology.nunique() == 1
            and not group.conflicting_sheets.any())
        aggregate_pattern = (reported.sum() == 1
            and carrier.summer_MW.iloc[0] > carrier.nameplate_MW.iloc[0] + 1.0)
        identified = consistent and aggregate_pattern
        ledger.loc[group.index, 'shared_capacity_ambiguous'] = not identified
        if identified:
            ledger.loc[group.index[missing], 'shared_capacity_member'] = True
            ledger.loc[group.index[reported], 'aggregate_capacity_carrier'] = True
        audit.append(dict(snapshot_year=year, plant_id=plant, unit_code=code,
            plant_name=group.plant_name.iloc[0], region=group.region.iloc[0],
            members=len(group), missing_members=int(missing.sum()),
            reported_summer_MW=group.summer_MW.sum(), nameplate_MW=group.nameplate_MW.sum(),
            identified_reporting_pattern=identified, consistent_status_and_BA=consistent,
            carrier_unit_keys=';'.join(carrier.unit_key),
            blank_unit_keys=';'.join(group.loc[missing, 'unit_key']),
            meaning='reporting allocation only; no individual CT capacity inferred'))
    return ledger, pd.DataFrame(audit)


def load_inventory():
    rows, manifests = [], []
    for year in range(2019, 2027):
        path = source_path(year)
        manifests.append(dict(year=year, file=str(path.relative_to(HERE.parent)), sha256=sha(path),
            url=f'https://www.eia.gov/electricity/data/eia860m/archive/xls/june_generator{year}.xlsx',
            assumed_information_cutoff=f'December {year}; June archive release bytes not independently certified'))
        for sheet in SHEETS:
            frame = parser.read_860m(path, sheet)
            if frame.unit_key.duplicated().any():
                raise ValueError(f'Duplicate unit in {path}:{sheet}')
            out = pd.DataFrame(dict(snapshot_year=year, sheet=sheet, unit_key=frame.unit_key,
                plant_id=frame['Plant ID'], BA=frame['Balancing Authority Code'].astype(str).str.strip(),
                plant_name=frame.get('Plant Name', pd.Series('', index=frame.index)),
                unit_code=frame.get('Unit Code', pd.Series('', index=frame.index)).fillna('').astype(str).str.strip(),
                prime_mover=frame.get('Prime Mover Code', pd.Series('', index=frame.index)).fillna('').astype(str).str.strip(),
                sector=frame.get('Sector', pd.Series('', index=frame.index)),
                status=frame.get('Status', pd.Series('', index=frame.index)).map(parser.status_code),
                technology=frame.apply(technology, axis=1)))
            out['region'] = out.BA.map(parser.BA_TO_REGION)
            for field, source in {
                'nameplate_MW': 'Nameplate Capacity (MW)', 'summer_MW': 'Net Summer Capacity (MW)',
                'operating_year': 'Operating Year', 'operating_month': 'Operating Month',
                'retirement_year': 'Planned Retirement Year', 'retirement_month': 'Planned Retirement Month',
                'operation_year': 'Planned Operation Year', 'operation_month': 'Planned Operation Month',
                'uprate_year': 'Planned Uprate Year', 'uprate_month': 'Planned Uprate Month',
                'uprate_MW': 'Planned Uprate of Summer Capacity (MW)',
                'derate_year': 'Planned Derate Year', 'derate_month': 'Planned Derate Month',
                'derate_MW': 'Planned Derate of Summer Capacity (MW)',
            }.items():
                out[field] = numeric(frame, source)
            counts = frame.unit_key.value_counts()
            out['within_sheet_duplicate'] = out.unit_key.map(counts).gt(1)
            rows.append(out)
        print(f'Parsed June {year}', flush=True)
    ledger = pd.concat(rows, ignore_index=True)
    n = ledger.groupby(['snapshot_year', 'unit_key']).sheet.transform('nunique')
    ledger['conflicting_sheets'] = n.gt(1)
    ledger['is_active_OP'] = ledger.sheet.eq('Operating') & ledger.status.eq('OP') & ~ledger.conflicting_sheets
    ledger['known_capacity_MW'] = ledger.summer_MW.where(ledger.summer_MW.ge(0))
    ledger, shared_audit = identify_shared_capacity(ledger)
    ledger['capacity_for_fit_MW'] = ledger.summer_MW.where(ledger.summer_MW.gt(0), ledger.nameplate_MW)
    # Covered blanks carry no additional capacity weight, not zero engineering capability.
    ledger.loc[ledger.shared_capacity_member, 'capacity_for_fit_MW'] = 0.0
    unresolved_cc = ledger.prime_mover.isin(['CA', 'CT', 'CS', 'CC']) & ledger.summer_MW.isna()
    ledger.loc[unresolved_cc, 'capacity_for_fit_MW'] = 0.0
    shared_audit.to_csv(HERE/'32_shared_capacity_reporting_audit.csv', index=False)
    return ledger, pd.DataFrame(manifests)


def read_peak_weights():
    labels = {'MISO': 'MISO', 'PJM': 'PJM', 'SPP': 'SPP', 'MRO-SPP': 'SPP',
              'Texas RE-ERCOT': 'ERCOT', 'NPCC-New York': 'NYISO', 'NPCC-New England': 'ISO-NE'}
    rows = []
    for year in range(2020, 2027):
        path = INPUTS / f'nerc_sra/nerc_sra_{year}.pdf'
        reader = PdfReader(path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ''
            if not ('Nameplate' in text and ('Generation Resources by' in text or 'Energy Resources by' in text)):
                continue
            for line in text.splitlines():
                line = re.sub(r'\s+', ' ', line.strip())
                for label, region in labels.items():
                    if not line.startswith(label + ' '):
                        continue
                    tokens = line[len(label):].strip().split()
                    if len(tokens) not in [6, 9, 12]:
                        raise ValueError((year, index + 1, line, len(tokens)))
                    for j, tech in enumerate(['wind', 'solar']):
                        a, b, percent = tokens[j*3:j*3+3]
                        nameplate = float(a.replace(',', ''))
                        contribution = 0.0 if b in ['-', '–'] else float(b.replace(',', ''))
                        if nameplate <= 0:
                            raise ValueError('No source ratio exists for this technology')
                        rows.append(dict(year=year, region=region, technology=tech,
                            source_nameplate_MW=nameplate, source_expected_MW=contribution,
                            weight=contribution/nameplate, source_printed_pct=float(percent.rstrip('%')),
                            pdf_page=index+1, source_file=str(path.relative_to(HERE.parent)), sha256=sha(path),
                            capacity_basis='source peak-demand-hour contribution, not risk-hour ELCC'))
    table = pd.DataFrame(rows)
    if table.duplicated(['year', 'region', 'technology']).any() or len(table) != 84:
        raise ValueError('Incomplete or duplicate origin weight coverage')
    if not table.weight.between(0, 1).all():
        raise ValueError('Invalid wind/solar contribution ratios')
    return table


def resolve_target(frame):
    valid = frame[~frame.conflicting_sheets].copy()
    valid['target_active'] = np.where(valid.sheet.eq('Operating'), valid.status.eq('OP').astype(float), 0.0)
    invalid_status = valid.sheet.eq('Operating') & ~valid.status.isin(['OP', 'SB', 'OA', 'OS'])
    invalid_status |= valid.sheet.eq('Planned') & valid.status.eq('OP')
    valid.loc[invalid_status, 'target_active'] = np.nan
    return valid.set_index('unit_key')


def build_cohorts(ledger, config):
    blocks, unseen, accounting, coverage = [], [], [], []
    snapshots = {year: ledger[ledger.snapshot_year.eq(year)].copy() for year in range(2019, 2027)}
    for vintage in config['origins']:
        origin_all = snapshots[vintage]
        origin = origin_all[origin_all.region.notna() & origin_all.sheet.isin(['Operating', 'Planned']) & ~origin_all.conflicting_sheets].copy()
        current = origin_all[origin_all.is_active_OP & origin_all.region.notna()]
        complete = current[current.summer_MW.notna() & current.summer_MW.ge(0)
            & current.nameplate_MW.gt(0) & ~current.aggregate_capacity_carrier]
        ratios = complete.groupby('technology')[['summer_MW', 'nameplate_MW']].sum()
        ratios['ratio'] = ratios.summer_MW / ratios.nameplate_MW
        ratio_map = ratios.ratio.to_dict()
        origin['forecast_capacity_imputed'] = origin.summer_MW.isna() | origin.summer_MW.lt(0)
        origin.loc[origin.shared_capacity_member, 'forecast_capacity_imputed'] = False
        blocked_cc = origin.prime_mover.isin(['CA', 'CT', 'CS', 'CC']) & origin.summer_MW.isna()
        origin.loc[blocked_cc, 'forecast_capacity_imputed'] = False
        estimated = origin.nameplate_MW * origin.technology.map(ratio_map).fillna(1.0)
        origin['forecast_unit_MW'] = origin.summer_MW.where(~origin.forecast_capacity_imputed, estimated)
        origin.loc[origin.shared_capacity_member, 'forecast_unit_MW'] = 0.0
        origin.loc[blocked_cc & ~origin.shared_capacity_member, 'forecast_unit_MW'] = np.nan
        # Observations retain missing capacity; only forecast features are estimated.
        for horizon in config['horizons']:
            target_year = vintage + horizon
            if target_year > config['target_end']:
                continue
            target_all = snapshots[target_year]
            target = resolve_target(target_all)
            part = origin.copy()
            part['vintage'], part['target_year'], part['horizon'] = vintage, target_year, horizon
            part['retirement_due'] = due(part.retirement_year, part.retirement_month, target_year)
            part['operation_due'] = due(part.operation_year, part.operation_month, target_year)
            part['origin_state'] = np.where(part.sheet.eq('Planned'), part.status.map(parser.stage_group), part.status)
            part['schedule_due'] = np.where(part.sheet.eq('Planned'), part.operation_due, part.retirement_due)
            part['target_active'] = part.unit_key.map(target.target_active)
            part['target_region'] = part.unit_key.map(target.region)
            part['target_summer_MW'] = part.unit_key.map(target.summer_MW)
            part['target_nameplate_MW'] = part.unit_key.map(target.nameplate_MW)
            part['target_technology'] = part.unit_key.map(target.technology)
            part['target_status'] = part.unit_key.map(target.status)
            part['same_BA_region'] = part.region.eq(part.target_region)
            part['observed_active_in_origin_region'] = part.target_active.where(part.target_active.isna(),
                part.target_active * part.same_BA_region.astype(float))
            part['scheduled_probability'] = np.where(part.sheet.eq('Planned'), part.operation_due.astype(float),
                (part.status.eq('OP') & ~part.retirement_due).astype(float))
            modifications = (part.uprate_MW.fillna(0) * due(part.uprate_year, part.uprate_month, target_year)
                - part.derate_MW.fillna(0) * due(part.derate_year, part.derate_month, target_year))
            modifications = modifications.where(part.sheet.eq('Operating'), 0.0)
            part['scheduled_unit_MW'] = np.maximum(0.0, part.forecast_unit_MW + modifications)
            blocks.append(part)
            target_OP = target_all[target_all.is_active_OP & target_all.region.notna()].copy()
            orig_ids = set(origin_all.unit_key)
            orig_active_ids = set(origin_all[origin_all.is_active_OP].unit_key)
            visible_planned = set(origin_all[origin_all.sheet.eq('Planned')].unit_key)
            target_OP['seen_at_origin_any_sheet'] = target_OP.unit_key.isin(orig_ids)
            target_OP['operating_after_origin'] = ((target_OP.operating_year > vintage)
                | (target_OP.operating_year.eq(vintage) & target_OP.operating_month.gt(6)))
            target_OP['unseen_new'] = ~target_OP.seen_at_origin_any_sheet & target_OP.operating_after_origin
            target_OP['unseen_prior_or_undated'] = ~target_OP.seen_at_origin_any_sheet & ~target_OP.operating_after_origin
            target_OP['origin_active'] = target_OP.unit_key.isin(orig_active_ids)
            target_OP['origin_planned'] = target_OP.unit_key.isin(visible_planned)
            origin_region_map = current.set_index('unit_key').region
            target_OP['origin_region'] = target_OP.unit_key.map(origin_region_map)
            target_OP['BA_moved'] = target_OP.origin_active & target_OP.origin_region.ne(target_OP.region)
            for region in config['regions']:
                o = current[current.region.eq(region)].set_index('unit_key')
                t = target_OP[target_OP.region.eq(region)].set_index('unit_key')
                common = o.index.intersection(t.index)
                exiting = o.index.difference(t.index)
                entering = t.index.difference(o.index)
                survivor_size = (t.loc[common].summer_MW.fillna(0) - o.loc[common].summer_MW.fillna(0)).sum()
                removed = o.loc[exiting].summer_MW.sum()
                added = t.loc[entering].summer_MW.sum()
                delta = t.summer_MW.sum() - o.summer_MW.sum()
                accounting.append(dict(vintage=vintage, target_year=target_year, horizon=horizon, region=region,
                    origin_known_summer_MW=o.summer_MW.sum(), target_known_summer_MW=t.summer_MW.sum(),
                    survivor_reported_size_change_MW=survivor_size, exiting_capacity_MW=removed,
                    entering_capacity_MW=added, residual_MW=delta-(survivor_size-removed+added),
                    unseen_new_target_MW=t.loc[t.unseen_new].summer_MW.sum(),
                    unseen_prior_or_undated_MW=t.loc[t.unseen_prior_or_undated].summer_MW.sum(),
                    BA_inflow_MW=t.loc[t.BA_moved].summer_MW.sum(),
                    observed_visible_planned_entry_MW=t.loc[t.origin_planned].summer_MW.sum()))
                coverage.append(dict(vintage=vintage, target_year=target_year, region=region,
                    target_active_units=len(t), target_missing_summer_units=int(t.summer_MW.isna().sum()),
                    target_missing_summer_nameplate_MW=t.loc[t.summer_MW.isna()].nameplate_MW.sum(),
                    shared_reporting_blank_units=int(t.shared_capacity_member.sum()),
                    unresolved_missing_summer_units=int((t.summer_MW.isna() & ~t.shared_capacity_member).sum()),
                    unresolved_missing_nameplate_MW=t.loc[t.summer_MW.isna() & ~t.shared_capacity_member].nameplate_MW.sum(),
                    target_total_nameplate_MW=t.nameplate_MW.sum()))
                for tech in sorted(ledger.technology.unique()):
                    group = t[t.technology.eq(tech)]
                    unseen.append(dict(vintage=vintage, target_year=target_year, horizon=horizon, region=region,
                        technology=tech, unseen_new_summer_MW=group.loc[group.unseen_new].summer_MW.sum(),
                        unseen_new_nameplate_MW=group.loc[group.unseen_new].nameplate_MW.sum()))
        print(f'Built origin {vintage} cohorts', flush=True)
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(unseen), pd.DataFrame(accounting), pd.DataFrame(coverage)


def cell_probability(train, config):
    known = train[train.target_active.notna() & train.capacity_for_fit_MW.gt(0)].copy()
    if (known.unit_key.nunique() < config['minimum_cell_units']
            or known.plant_id.nunique() < config['minimum_cell_plants']
            or known.capacity_for_fit_MW.sum() < config['minimum_cell_capacity_MW']):
        return None
    repeats = known.groupby(['unit_key', 'horizon', 'target_year']).unit_key.transform('size')
    weights = known.capacity_for_fit_MW / repeats
    return float(np.clip(np.dot(weights, known.target_active) / weights.sum(), 0.0, 1.0))


def predict_visible(cohorts, vintage, config, heldout=None):
    test = cohorts[cohorts.vintage.eq(vintage)].copy()
    train = cohorts[(cohorts.vintage < vintage) & (cohorts.target_year <= vintage)].copy()
    if heldout:
        train = train[train.region.ne(heldout)]
    test['learned_probability'] = test.scheduled_probability.astype(float)
    test['fit_level'] = 'schedule_fallback'
    parameter_rows = []
    cell_keys = ['horizon', 'origin_state', 'technology', 'schedule_due']
    for key, cell in test.groupby(cell_keys, dropna=False):
        h, state, tech, scheduled = key
        pool = train[train.horizon.eq(h) & train.origin_state.eq(state) & train.schedule_due.eq(scheduled)]
        exact = pool[pool.technology.eq(tech)]
        p = cell_probability(exact, config)
        used = exact
        level = 'technology_state_horizon_schedule'
        if p is None:
            p = cell_probability(pool, config)
            used, level = pool, 'pooled_technology_state_horizon_schedule'
        if p is not None:
            test.loc[cell.index, 'learned_probability'] = p
            test.loc[cell.index, 'fit_level'] = level
        parameter_rows.append(dict(vintage=vintage, horizon=h, origin_state=state, technology=tech,
            schedule_due=scheduled, probability=np.nan if p is None else p, fit_level=level if p is not None else 'schedule_fallback',
            train_units=used.unit_key.nunique(), train_plants=used.plant_id.nunique(),
            max_training_origin=used.vintage.max() if len(used) else np.nan,
            max_training_target=used.target_year.max() if len(used) else np.nan,
            heldout_region=heldout))
    return test, pd.DataFrame(parameter_rows)


def peak_unit_values(frame, weights, origin, use_target=False):
    tech = frame.target_technology if use_target else frame.technology
    size = frame.target_summer_MW if use_target else frame.forecast_unit_MW
    np_size = frame.target_nameplate_MW if use_target else frame.nameplate_MW
    value = size.copy()
    source_weights = weights[weights.year.eq(origin)]
    lookup = source_weights.set_index(['region', 'technology']).weight
    for group in ['wind', 'solar']:
        mask = tech.eq(group)
        c = frame.region.map(lookup.xs(group, level='technology').to_dict())
        value.loc[mask] = np_size[mask] * c[mask]
    value.loc[tech.eq('storage')] = 0.0
    value.loc[frame.region.eq('CAISO')] = np.nan
    return value


def target_index(ledger, weights, origin, target, region):
    active = ledger[ledger.snapshot_year.eq(target) & ledger.is_active_OP & ledger.region.eq(region)].copy()
    active['forecast_unit_MW'] = active.summer_MW
    return peak_unit_values(active, weights, origin).sum(min_count=1)


def aggregate_predictions(cohorts, unseen, ledger, weights, config, heldout=None):
    rows, params, unit_preds = [], [], []
    for vintage in config['origins']:
        forecast, parameters = predict_visible(cohorts, vintage, config, heldout)
        params.append(parameters)
        forecast['heldout_region'] = heldout
        unit_preds.append(forecast)
        for (target, region), group in forecast.groupby(['target_year', 'region']):
            h = int(target-vintage)
            observed = ledger[ledger.snapshot_year.eq(target) & ledger.is_active_OP & ledger.region.eq(region)]
            observed_origin = ledger[ledger.snapshot_year.eq(vintage) & ledger.is_active_OP & ledger.region.eq(region)]
            target_capacity = observed.summer_MW.sum()
            origin_capacity = observed_origin.summer_MW.sum()
            index_available = vintage >= 2020 and region != 'CAISO'
            index_units = peak_unit_values(group, weights, vintage) if index_available else pd.Series(np.nan, index=group.index)
            index_target = target_index(ledger, weights, vintage, target, region) if index_available else np.nan
            index_origin = target_index(ledger, weights, vintage, vintage, region) if index_available else np.nan
            background_capacity = background_index = 0.0
            supported_cells = 0
            for tech in sorted(ledger.technology.unique()):
                history = unseen[(unseen.vintage < vintage) & (unseen.target_year <= vintage)
                    & unseen.region.eq(region) & unseen.technology.eq(tech) & unseen.horizon.eq(h)]
                if history.vintage.nunique() < 2:
                    continue
                supported_cells += 1
                physical = float(history.unseen_new_summer_MW.mean())
                background_capacity += physical
                if index_available:
                    if tech in ['wind', 'solar']:
                        c = weights[weights.year.eq(vintage) & weights.region.eq(region) & weights.technology.eq(tech)].weight.item()
                        background_index += float(history.unseen_new_nameplate_MW.mean()) * c
                    elif tech != 'storage':
                        background_index += physical
            for model in MODELS:
                if model == 'inventory_persistence':
                    cap, index = origin_capacity, index_origin
                elif model == 'announced_schedule':
                    cap = float((group.scheduled_unit_MW * group.scheduled_probability).sum())
                    # Modifications are thermal/hydro physical size, not a new wind/solar nameplate forecast.
                    adjustment = (group.scheduled_unit_MW - group.forecast_unit_MW).where(~group.technology.isin(['wind','solar','storage']),0)
                    index = float(((index_units + adjustment) * group.scheduled_probability).sum()) if index_available else np.nan
                else:
                    cap = float((group.forecast_unit_MW * group.learned_probability).sum())
                    index = float((index_units * group.learned_probability).sum()) if index_available else np.nan
                    if model == 'learned_open_stock':
                        cap += background_capacity
                        index += background_index
                rows.append(dict(vintage=vintage, target_year=target, horizon=h, region=region, model=model,
                    predicted_summer_MW=cap, observed_summer_MW=target_capacity,
                    predicted_index_MW=index, observed_index_MW=index_target,
                    inventory_scope='EIA BA OP stock; no NERC footprint substitution',
                    index_available=index_available, learned_capacity_share=float(group.loc[group.fit_level.ne('schedule_fallback')].forecast_unit_MW.sum()/group.forecast_unit_MW.sum()),
                    unknown_label_origin_capacity_MW=group.loc[group.target_active.isna()].forecast_unit_MW.sum(),
                    imputed_origin_capacity_MW=group.loc[group.forecast_capacity_imputed].forecast_unit_MW.sum(),
                    shared_reporting_blank_units=int(group.shared_capacity_member.sum()),
                    unresolved_origin_capacity_units=int(group.forecast_unit_MW.isna().sum()),
                    expected_unseen_summer_MW=background_capacity if model == 'learned_open_stock' else 0,
                    background_supported_technology_cells=supported_cells, heldout_region=heldout))
        print(f'Predicted origin {vintage}, heldout={heldout}', flush=True)
    return pd.DataFrame(rows), pd.concat(params, ignore_index=True), pd.concat(unit_preds, ignore_index=True)


def metrics(predictions):
    rows = []
    for scope, pool in [('all', predictions), ('short_horizon_supported_origins', predictions[(predictions.vintage >= 2022)&(predictions.horizon<=2)])]:
        for split in ['overall', 'region', 'horizon', 'vintage']:
            groups = pool.groupby('model') if split == 'overall' else pool.groupby(['model', split])
            for key, g in groups:
                model, label = (key, 'all') if split == 'overall' else key
                for quantity, pred, obs in [('physical_summer_stock','predicted_summer_MW','observed_summer_MW'),
                                           ('origin_weighted_capacity_index','predicted_index_MW','observed_index_MW')]:
                    valid = g[g[pred].notna() & g[obs].notna()]
                    error = (valid[pred]-valid[obs])/1000
                    rows.append(dict(scope=scope, split=split, subgroup=label, model=model, quantity=quantity,
                        cases=len(valid), outcome_region_targets=valid[['region','target_year']].drop_duplicates().shape[0],
                        MAE_GW=error.abs().mean(), bias_GW=error.mean(), RMSE_GW=np.sqrt(np.square(error).mean())))
    return pd.DataFrame(rows)


def diagnostics(predictions, config):
    paired, deletions, intervals = [], [], []
    rng = np.random.default_rng(config['seed'])
    for scope, pool in [('all', predictions), ('short_horizon_supported_origins', predictions[(predictions.vintage>=2022)&(predictions.horizon<=2)])]:
        for model in ['learned_visible_stock','learned_open_stock']:
            for baseline in ['inventory_persistence','announced_schedule']:
                for quantity, prediction, observation in [('physical_summer_stock','predicted_summer_MW','observed_summer_MW'),
                                                        ('origin_weighted_capacity_index','predicted_index_MW','observed_index_MW')]:
                    g = pool[pool.model.eq(model)].merge(pool[pool.model.eq(baseline)][KEYS+[prediction]].rename(columns={prediction:'baseline_prediction'}),on=KEYS,validate='one_to_one')
                    g = g[g[prediction].notna() & g.baseline_prediction.notna()].copy()
                    g['improvement_GW'] = ((g.baseline_prediction-g[observation]).abs()-(g[prediction]-g[observation]).abs())/1000
                    info = dict(scope=scope, model=model, baseline=baseline, quantity=quantity)
                    paired.append(g[KEYS+['improvement_GW']].assign(**info))
                    for dimension in ['region','vintage']:
                        for omitted in sorted(g[dimension].unique()):
                            deletions.append(dict(**info, dimension=dimension, omitted=omitted,
                                improvement_GW=g.loc[g[dimension].ne(omitted)].improvement_GW.mean()))
                    for dimension in ['region','target_year']:
                        blocks = g.groupby(dimension).improvement_GW.agg(['sum','count'])
                        idx = rng.integers(0,len(blocks),(config['bootstrap_draws'],len(blocks)))
                        values = blocks['sum'].to_numpy()[idx].sum(axis=1)/blocks['count'].to_numpy()[idx].sum(axis=1)
                        intervals.append(dict(**info, dimension=dimension, blocks=len(blocks),
                            improvement_GW=g.improvement_GW.mean(), ci_low_GW=np.quantile(values,.025),ci_high_GW=np.quantile(values,.975)))
    return pd.concat(paired,ignore_index=True),pd.DataFrame(deletions),pd.DataFrame(intervals)


def main():
    config = json.loads((HERE/'protocol.json').read_text())
    protected = list(INPUTS.rglob('*')) + list((HERE.parent/'expected/inventory').glob('*.csv'))
    before = {str(p.relative_to(HERE.parent)):sha(p) for p in protected if p.is_file()}
    ledger, source_manifest = load_inventory()
    ledger.to_csv(HERE/'01_full_inventory_ledger.csv',index=False)
    source_manifest.to_csv(HERE/'02_EIA_source_manifest.csv',index=False)
    weights = read_peak_weights()
    weights.to_csv(HERE/'03_source_peak_contribution_weights.csv',index=False)
    cohorts, unseen, flow, coverage = build_cohorts(ledger,config)
    cohorts.to_csv(HERE/'04_target_active_cohorts.csv',index=False)
    unseen.to_csv(HERE/'05_unseen_entry_cohorts.csv',index=False)
    flow.to_csv(HERE/'06_exact_observed_stock_flow.csv',index=False)
    coverage.to_csv(HERE/'07_capacity_missingness.csv',index=False)
    predictions, parameters, units = aggregate_predictions(cohorts,unseen,ledger,weights,config)
    predictions.to_csv(HERE/'08_predictions.csv',index=False)
    parameters.to_csv(HERE/'09_probability_parameters.csv',index=False)
    units[['vintage','target_year','horizon','region','unit_key','technology','origin_state','schedule_due',
        'target_active','learned_probability','fit_level','forecast_unit_MW','forecast_capacity_imputed',
        'shared_capacity_member','aggregate_capacity_carrier','shared_capacity_ambiguous',
        'target_region','target_status','same_BA_region','heldout_region']].to_csv(HERE/'10_unit_predictions.csv',index=False)
    metrics(predictions).to_csv(HERE/'11_metrics.csv',index=False)
    paired, deletion, intervals = diagnostics(predictions,config)
    paired.to_csv(HERE/'12_paired_improvement.csv',index=False)
    deletion.to_csv(HERE/'13_deletion_diagnostics.csv',index=False)
    intervals.to_csv(HERE/'14_block_intervals.csv',index=False)
    status = ledger[ledger.region.notna()].groupby(['snapshot_year','region','sheet','status'],dropna=False).agg(
        units=('unit_key','size'),reported_summer_MW=('summer_MW','sum'),nameplate_MW=('nameplate_MW','sum')).reset_index()
    status.to_csv(HERE/'15_status_inclusion_audit.csv',index=False)
    qa = [dict(check='exact_stock_flow_identity',passed=flow.residual_MW.abs().max()<1e-6,value=flow.residual_MW.abs().max()),
          dict(check='probabilities_bounded',passed=units.learned_probability.between(0,1).all(),value=len(units)),
          dict(check='no_future_training_target',passed=(parameters.max_training_target.dropna()<=parameters.loc[parameters.max_training_target.notna(),'vintage']).all(),value=len(parameters)),
          dict(check='no_same_origin_training',passed=(parameters.max_training_origin.dropna()<parameters.loc[parameters.max_training_origin.notna(),'vintage']).all(),value=len(parameters)),
          dict(check='unknown_labels_remain_nan',passed=cohorts.loc[cohorts.target_status.isna(),'target_active'].isna().all(),value=int(cohorts.target_active.isna().sum())),
          dict(check='nonnegative_predictions',passed=predictions.predicted_summer_MW.ge(0).all(),value=len(predictions)),
          dict(check='fixed_all_case_count',passed=len(predictions)==700,value=len(predictions)),
          dict(check='forecast_weight_max_year_is_origin',passed=(weights.year.min()==2020),value=weights.year.min()),
          dict(check='old_files_preserved',passed=all(sha(HERE.parent/p)==h for p,h in before.items()),value=len(before))]
    qa.append(dict(check='shared_reporting_blanks_not_imputed',
        passed=units.loc[units.shared_capacity_member,'forecast_unit_MW'].eq(0).all(),
        value=int(units.shared_capacity_member.sum())))
    pd.DataFrame(qa).to_csv(HERE/'16_QA.csv',index=False)
    pd.DataFrame([dict(file=p,before_sha256=h,after_sha256=sha(HERE.parent/p)) for p,h in before.items()]).to_csv(HERE/'17_preservation_manifest.csv',index=False)
    (HERE/'run_record.json').write_text(json.dumps(dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        protocol_sha256=sha(HERE/'protocol.json'),builder_sha256=sha(Path(__file__)),
        cohort_rows=len(cohorts),unknown_labels=int(cohorts.target_active.isna().sum()),source_peak_weights=len(weights),
        outcome_scope='physical June OP stock and fixed-origin index only; not full native balance validation',
        implementation_repairs='EIA shared-capacity reporting carriers retained; covered blanks have no extra capacity or fit weight; unresolved CC blanks remain unknown; independent-row imputation ratio excludes aggregate carriers; NS/NC is not artificially capped at 1',
        all_QA_passed=bool(pd.DataFrame(qa).passed.all())),indent=2))
    if not pd.DataFrame(qa).passed.all():
        raise AssertionError('QA failure; inspect 16_QA.csv')
    print(metrics(predictions).query("scope=='all' and split=='overall'").to_string(index=False),flush=True)


if __name__=='__main__':
    main()
