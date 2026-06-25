# Dataset Index

Built 11 stochastic instances. Each has instance.json + scenarios.npz (q[S,N], tt[S,N,N,3], feas[S,N,N,3], dist[N,N]) + meta.json.

| instance      |   nodes |   scenarios | speeds_kmh     |   demand_mean_m3 |   demand_cv |   feas_low |   feas_high |
|:--------------|--------:|------------:|:---------------|-----------------:|------------:|-----------:|------------:|
| dublin_real   |     101 |          60 | 21.0/25.4/31.0 |             1.2  |        0.74 |       0.91 |        0.26 |
| nyc_brooklyn  |      31 |          60 | 12.2/13.9/16.3 |             1.43 |        0.52 |       0.91 |        0.51 |
| nyc_manhattan |      61 |          60 | 14.3/17.8/23.1 |             1.09 |        0.52 |       0.86 |        0.26 |
| nyc_queens    |      31 |          60 | 27.8/36.9/46.9 |             1.2  |        0.52 |       0.82 |        0.2  |
| peshawar_real |     111 |          60 | 18.1/21.9/26.7 |             1.55 |        0.56 |       0.92 |        0.3  |
| solomon_c101  |     101 |          60 | 32.0/38.8/47.6 |            18.07 |        0.73 |       0.81 |        0.18 |
| solomon_c201  |     101 |          60 | 31.6/38.7/47.3 |            18.11 |        0.7  |       0.84 |        0.22 |
| solomon_r102  |     101 |          60 | 31.7/38.7/46.8 |            14.71 |        0.77 |       0.83 |        0.21 |
| solomon_r202  |     101 |          60 | 32.0/38.7/46.7 |            14.48 |        0.72 |       0.84 |        0.27 |
| solomon_rc101 |     101 |          60 | 31.7/38.7/46.7 |            17.15 |        0.66 |       0.82 |        0.2  |
| solomon_rc201 |     101 |          60 | 32.1/38.8/46.9 |            17.31 |        0.7  |       0.83 |        0.25 |

## Provenance

- **dublin_real**: geom=REAL Fingal solar-bin coordinates (190 bins, dense subset); tt=REAL-CALIBRATED distance x NYC empirical congestion-by-hour regime; demand=REAL per-bin Liters level x REAL-CALIBRATED CV (Austin)
- **nyc_brooklyn**: geom=REAL NYC TLC zone centroids; tt=REAL NYC TLC OD x hour; demand=REAL borough level (DSNY) x REAL-CALIBRATED CV (Austin)
- **nyc_manhattan**: geom=REAL NYC TLC zone centroids; tt=REAL NYC TLC OD x hour; demand=REAL borough level (DSNY) x REAL-CALIBRATED CV (Austin)
- **nyc_queens**: geom=REAL NYC TLC zone centroids; tt=REAL NYC TLC OD x hour; demand=REAL borough level (DSNY) x REAL-CALIBRATED CV (Austin)
- **peshawar_real**: geom=REAL WSSP Zone D coordinates (manuscript register); tt=REAL-CALIBRATED distance x NYC congestion; demand=REAL per-node demand (m3) x REAL-CALIBRATED CV (Austin)
- **solomon_c101**: geom=BENCHMARK Solomon (classical); tt=REAL-CALIBRATED distance x NYC congestion regime; demand=BENCHMARK level x REAL-CALIBRATED CV (Austin)
- **solomon_c201**: geom=BENCHMARK Solomon (classical); tt=REAL-CALIBRATED distance x NYC congestion regime; demand=BENCHMARK level x REAL-CALIBRATED CV (Austin)
- **solomon_r102**: geom=BENCHMARK Solomon (classical); tt=REAL-CALIBRATED distance x NYC congestion regime; demand=BENCHMARK level x REAL-CALIBRATED CV (Austin)
- **solomon_r202**: geom=BENCHMARK Solomon (classical); tt=REAL-CALIBRATED distance x NYC congestion regime; demand=BENCHMARK level x REAL-CALIBRATED CV (Austin)
- **solomon_rc101**: geom=BENCHMARK Solomon (classical); tt=REAL-CALIBRATED distance x NYC congestion regime; demand=BENCHMARK level x REAL-CALIBRATED CV (Austin)
- **solomon_rc201**: geom=BENCHMARK Solomon (classical); tt=REAL-CALIBRATED distance x NYC congestion regime; demand=BENCHMARK level x REAL-CALIBRATED CV (Austin)
