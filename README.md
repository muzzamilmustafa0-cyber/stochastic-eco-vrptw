# Stochastic Emission-Capacitated VRPTW with Eco-Speed

Code and data for a stochastic, learning-augmented eco-routing study: an
Emission-Capacitated Vehicle Routing Problem with Time Windows where bin-fill demand
and arc travel times are uncertain. Routes, service sequence, and discrete eco-speed
levels are optimised against expected fuel, emissions, and recourse cost, with
capacity and time windows handled through chance constraints and recourse.

## Layout

```
model/                     solver and learning code
  ecvrptw.py               problem model, fuel/emission model, scenario evaluator
  construct.py             nearest-feasible constructor
  hgls.py                  scenario-aware Hybrid Guided Local Search
  predictors.py           travel-time and eco-speed feasibility models (NYC TLC)
  decision_focus.py        decision-focused chance-constraint planning
  experiments.py           baseline methods, out-of-sample protocol
  run_campaign.py          full method x instance x seed runner (resumable)
  ablation.py              component ablation
  sensitivity.py           recourse-weight sensitivity
  analyze.py               aggregation, Wilcoxon/Friedman tests
  figures.py               result figures

data_pipeline/
  scripts/                 dataset construction (01 travel time ... 07 demand model)
  instances/               built instance families (instance.json + scenarios.npz)
  processed/               learned distributions and model reports
  DATA_MANIFEST.md         data sources and provenance
  DATASET_INDEX.md         instance summary

results/                   campaign, ablation, sensitivity outputs and figures
```

## Data

Travel-time uncertainty is built from NYC TLC yellow and green taxi records;
demand variability from Austin waste-collection loads and NYC DSNY tonnage;
geometry from NYC taxi zones, Fingal (Dublin) bin locations, the Peshawar WSSP
Zone D register, and Solomon benchmark instances. Sources and per-item provenance
are listed in `data_pipeline/DATA_MANIFEST.md`.

Raw downloads are not tracked (see `.gitignore`); they are reproduced by the
numbered scripts in `data_pipeline/scripts/`. The built instances under
`data_pipeline/instances/` are sufficient to run all experiments.

## Reproduce

```
pip install -r requirements.txt

# rebuild datasets from raw (optional; instances are already provided)
python data_pipeline/scripts/01_travel_time_distributions.py
python data_pipeline/scripts/02_demand_distributions.py
python data_pipeline/scripts/03_fuse_nyc_instances.py
python data_pipeline/scripts/04_fuse_dublin_instance.py
python data_pipeline/scripts/05_fuse_solomon_benchmark.py
python data_pipeline/scripts/06_fuse_peshawar_instance.py

# experiments
python -m model.run_campaign      # main comparison
python -m model.ablation          # ablation
python -m model.sensitivity       # recourse-weight sweep
python -m model.analyze           # tables and tests
python -m model.figures           # figures
```

Each instance is split into planning and held-out evaluation scenarios; every method
is scored on the held-out set. Long runs write one row per result and resume on restart.

## Reproduction map

One master sequence reproduces every reported number from a clean environment:

```
pip install -r requirements.txt
python data_pipeline/scripts/08_dependence_calibration.py   # dependence parameters
python data_pipeline/scripts/09_build_arc_lookups.py        # arc-speed lookups
python data_pipeline/scripts/10_prediction_validation.py    # calibration metrics (Fig on calibration)
python -m model.test_recourse                               # recourse unit tests
python -m model.run_final                                   # all eight experiment stages
python -m model.run_extras                                  # anytime traces, 30-seed repeats
python -m model.analyze2                                    # statistics tables
python -m model.figures2                                    # result figures
```

Script-to-output map: `run_final.py` writes `results2/main.csv` (main comparison
table and per-instance appendix), `exact.csv` (verification table),
`shifts.csv` (shift figure), `scount.csv` (scenario-count figure),
`ablation.csv` (component ladder figure), `sens.csv` (weight sensitivity),
`repeats.csv` and `repeats30.csv` (seed variability), `policy_theta.json`
(learned policy); `analyze2.py` writes the paired statistics; `figures2.py`
writes every figure to `results2/figures/`. Scenario replicates regenerate
deterministically from CRC-32 seeds of the family and replicate labels, so no
bulk scenario data is stored.

## Data sources

- New York City Taxi and Limousine Commission (2024). TLC trip record data,
  yellow and green trips, January, April, July 2024. Accessed 22 June 2026.
- City of Austin (2021). Waste collection and diversion report, daily loads,
  dataset mbnu-4wq9. Accessed 22 June 2026.
- New York City Department of Sanitation (2026). Monthly tonnage, dataset
  ebb7-mvp5. Accessed 22 June 2026.
- Fingal County Council (2024). Solar-compactor public litter bins. Accessed
  22 June 2026.
- Solomon (1987) 100-customer VRPTW benchmark instances.
- WSSP Peshawar Zone D register: administrative document; included only as the
  constructed routing instance.
