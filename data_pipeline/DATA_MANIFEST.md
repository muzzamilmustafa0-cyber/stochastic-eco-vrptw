# Data Manifest

Sources and provenance. Each item is tagged REAL (measured), REAL-CALIBRATED (real
variability statistics applied to nodes), or BENCHMARK (classical geometry).

## Travel-time uncertainty
| ID | Source | Type | Granularity | URL |
|----|--------|------|-------------|-----|
| TT-NYC | NYC TLC yellow + green taxi trip records | REAL | zone x zone x hour travel-time distribution | https://d37ci6vzurychx.cloudfront.net/trip-data/ |
| TT-ZONES | NYC TLC taxi zone lookup + shapefile | REAL | 263 zone centroids | https://d37ci6vzurychx.cloudfront.net/misc/ |

## Demand (waste) variability
| ID | Source | Type | Granularity | URL |
|----|--------|------|-------------|-----|
| DM-NYC | NYC DSNY monthly tonnage | REAL | community district x month | https://data.cityofnewyork.us/resource/ebb7-mvp5.csv |
| DM-AUS | Austin waste collection loads | REAL | per-route load weight, timestamped | https://data.austintexas.gov/resource/mbnu-4wq9.csv |
| DM-DUB | Fingal solar bins | REAL | per-bin volume + coordinates | https://data.fingal.ie/ |

## Geometry
| ID | Source | Type |
|----|--------|------|
| GEO-NYC | TLC taxi zone centroids | REAL |
| GEO-DUB | Fingal solar-bin coordinates | REAL |
| GEO-PESH | Peshawar WSSP Zone D register (manuscript appendix) | REAL |
| GEO-BENCH | Solomon instances | BENCHMARK |

## Not used
- Mendeley "Smart Bin Insights" (8wc9jtndf6): contains a document, not a numeric dataset.
- Chicago Taxi (Socrata wrvz-psew): endpoint repeatedly timed out.
- Fingal bin fill-level time series: not published (coordinates and volume only).

## Built instance families
See DATASET_INDEX.md / DATASET_INDEX.csv for sizes and parameters. Each family is
instance.json + scenarios.npz (q[S,N], tt[S,N,N,3], feas[S,N,N,3], dist[N,N]) + meta.json.

- nyc_manhattan, nyc_queens, nyc_brooklyn: TLC geometry + TLC travel time + DSNY demand
  level scaled by Austin coefficient of variation.
- dublin_real: Fingal coordinates and per-bin volume; travel time from the NYC
  congestion-by-hour profile.
- solomon_{c101,c201,r102,r202,rc101,rc201}: Solomon geometry and time windows with the
  stochastic eco layers added.
- peshawar_real: WSSP Zone D register with the stochastic eco layers added.

Yellow and green TLC cover Manhattan, Queens, and Brooklyn intra-borough; the Bronx is
omitted (sparse taxi coverage). Eco-speed levels are set per city from percentiles of the
observed achievable speed.
