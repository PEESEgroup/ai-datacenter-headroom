"""Unchanged mechanism contrasts from the existing SI analysis."""
from pathlib import Path
import numpy as np
import pandas as pd
KEYS = ['vintage', 'target_year', 'region']
UNIT_KEYS = [*KEYS, 'unit_key']
MODELS = ['existing_plan_project_frequency', 'existing_plan_project_frequency_open']
QUANTITIES = [('physical_summer_stock', 'predicted_summer_MW', 'observed_summer_MW'),
              ('origin_weighted_capacity_index', 'predicted_index_MW', 'observed_index_MW')]


def hybrid_units(cohorts, unit_predictions, weights, stock):
    out = cohorts.merge(unit_predictions[UNIT_KEYS + ['learned_probability', 'fit_level']],
                        on=UNIT_KEYS, validate='one_to_one')
    planned = out.sheet.eq('Planned')
    out['hybrid_probability'] = out.learned_probability.where(planned, out.scheduled_probability)
    out['hybrid_unit_MW'] = out.forecast_unit_MW.where(planned, out.scheduled_unit_MW)
    out['origin_index_unit_MW'] = np.nan
    out['target_index_unit_MW'] = np.nan
    for origin in sorted(out.vintage.unique()):
        mask = out.vintage.eq(origin)
        if origin >= 2020:
            out.loc[mask, 'origin_index_unit_MW'] = stock.peak_unit_values(out.loc[mask], weights, origin)
            out.loc[mask, 'target_index_unit_MW'] = stock.peak_unit_values(out.loc[mask], weights, origin, use_target=True)
    adjustment = (out.scheduled_unit_MW - out.forecast_unit_MW).where(
        ~planned & ~out.technology.isin(['wind', 'solar', 'storage']), 0)
    out['hybrid_index_unit_MW'] = out.origin_index_unit_MW + adjustment
    return out

def hybrid_predictions(units, old):
    visible = old[old.model.eq('learned_visible_stock')].set_index(KEYS)
    opened = old[old.model.eq('learned_open_stock')].set_index(KEYS)
    rows = []
    for keys, group in units.groupby(KEYS):
        origin, target, region = keys
        source = visible.loc[keys]
        cap = (group.hybrid_unit_MW * group.hybrid_probability).sum()
        index = (group.hybrid_index_unit_MW * group.hybrid_probability).sum() if source.index_available else np.nan
        cap_background = opened.loc[keys, 'predicted_summer_MW'] - source.predicted_summer_MW
        index_background = opened.loc[keys, 'predicted_index_MW'] - source.predicted_index_MW
        for model in MODELS:
            add = model.endswith('_open')
            rows.append(dict(vintage=origin, target_year=target, horizon=target-origin, region=region, model=model,
                predicted_summer_MW=cap + (cap_background if add else 0), observed_summer_MW=source.observed_summer_MW,
                predicted_index_MW=index + (index_background if add else 0), observed_index_MW=source.observed_index_MW,
                index_available=source.index_available, expected_unseen_summer_MW=cap_background if add else 0,
                expected_unseen_index_MW=index_background if add else 0,
                inventory_scope=source.inventory_scope,
                unknown_labels_not_converted_to_zero=True, native_balance_substitution_allowed=False))
    return pd.concat([old, pd.DataFrame(rows)], ignore_index=True)

def diagnostics(predictions, config):
    rows, deletions, intervals = [], [], []
    rng = np.random.default_rng(config['seed'])
    for scope, pool in [('all', predictions), ('short_horizon_supported_origins',
                predictions[predictions.vintage.ge(2022) & predictions.horizon.le(2)])]:
        for model in MODELS:
            for baseline in ['inventory_persistence', 'announced_schedule', 'learned_open_stock']:
                for quantity, pred, obs in QUANTITIES:
                    group = pool[pool.model.eq(model)].merge(
                        pool[pool.model.eq(baseline)][KEYS + [pred]].rename(columns={pred: 'baseline_prediction'}),
                        on=KEYS, validate='one_to_one')
                    group = group[group[pred].notna() & group.baseline_prediction.notna() & group[obs].notna()].copy()
                    group['improvement_GW'] = ((group.baseline_prediction-group[obs]).abs() -
                                             (group[pred]-group[obs]).abs()) / 1000
                    info = dict(scope=scope, model=model, baseline=baseline, quantity=quantity)
                    rows.append(group[KEYS + ['improvement_GW']].assign(**info))
                    for dimension in ['region', 'vintage']:
                        for omitted in sorted(group[dimension].unique()):
                            deletions.append(dict(**info, dimension=dimension, omitted=omitted,
                                improvement_GW=group.loc[group[dimension].ne(omitted)].improvement_GW.mean()))
                    for dimension in ['region', 'target_year']:
                        blocks = group.groupby(dimension).improvement_GW.agg(['sum', 'count'])
                        samples = rng.integers(0, len(blocks), (config['bootstrap_draws'], len(blocks)))
                        values = blocks['sum'].to_numpy()[samples].sum(axis=1) / blocks['count'].to_numpy()[samples].sum(axis=1)
                        intervals.append(dict(**info, dimension=dimension, blocks=len(blocks),
                            improvement_GW=group.improvement_GW.mean(), ci_low_GW=np.quantile(values, .025),
                            ci_high_GW=np.quantile(values, .975)))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(deletions), pd.DataFrame(intervals)

def gates(intervals, deletions):
    rows = []
    for (scope, model, quantity), group in intervals.groupby(['scope', 'model', 'quantity']):
        group = group[group.baseline.isin(['inventory_persistence', 'announced_schedule'])]
        deletion = deletions[deletions.scope.eq(scope) & deletions.model.eq(model) & deletions.quantity.eq(quantity)
                              & deletions.baseline.isin(['inventory_persistence', 'announced_schedule'])]
        mean = bool(group.improvement_GW.gt(0).all())
        remove = bool(deletion.improvement_GW.gt(0).all())
        ci = bool(group.loc[group.dimension.eq('region'), 'ci_low_GW'].gt(0).all())
        rows.append(dict(scope=scope, model=model, quantity=quantity, lower_MAE_than_both=mean,
            all_deletions_improve=remove, region_block_intervals_positive=ci,
            component_gate_passed=mean and remove and ci, native_full_balance_adoption_allowed=False))
    return pd.DataFrame(rows)
