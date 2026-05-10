# Cross-architecture XAI comparison

SG estimator: `gnn`  ·  ST path: `native`


## Per-cell summary

| regime   |   mt | hb   | sg_estimator   | st_path   |   rho_channel_rank |   jaccard_top5 |   jaccard_top10 |   rho_pair_offdiag |   n_sg_subjects |   n_sg_trials |   n_st_subjects |   n_st_trials |
|:---------|-----:|:-----|:---------------|:----------|-------------------:|---------------:|----------------:|-------------------:|----------------:|--------------:|----------------:|--------------:|
| kfold-5  |    2 | hbo  | gnn            | native    |              0.168 |          0.111 |           0.333 |              0.391 |              52 |            88 |              52 |            93 |
| kfold-5  |    2 | hbr  | gnn            | native    |             -0.089 |          0.111 |           0.250 |              0.261 |              52 |            88 |              50 |            92 |
| kfold-5  |    2 | hbt  | gnn            | native    |             -0.020 |          0.111 |           0.333 |              0.374 |              52 |            88 |              54 |            96 |
| kfold-5  |    4 | hbo  | gnn            | native    |             -0.176 |          0.111 |           0.333 |              0.251 |              46 |           149 |              56 |           182 |
| kfold-5  |    4 | hbr  | gnn            | native    |             -0.095 |          0.111 |           0.333 |              0.212 |              46 |           149 |              54 |           172 |
| kfold-5  |    4 | hbt  | gnn            | native    |             -0.019 |          0.111 |           0.333 |              0.303 |              46 |           149 |              56 |           177 |
| kfold-10 |    2 | hbo  | gnn            | native    |              0.090 |          0.250 |           0.250 |              0.352 |              56 |            97 |              54 |            95 |
| kfold-10 |    2 | hbr  | gnn            | native    |              0.134 |          0.250 |           0.250 |              0.309 |              56 |            97 |              54 |            95 |
| kfold-10 |    2 | hbt  | gnn            | native    |              0.008 |          0.111 |           0.176 |              0.325 |              56 |            97 |              53 |            96 |
| kfold-10 |    4 | hbo  | gnn            | native    |             -0.285 |          0.250 |           0.333 |              0.255 |              52 |           162 |              55 |           186 |
| kfold-10 |    4 | hbr  | gnn            | native    |             -0.345 |          0.000 |           0.250 |              0.212 |              52 |           162 |              57 |           183 |
| kfold-10 |    4 | hbt  | gnn            | native    |             -0.078 |          0.250 |           0.333 |              0.230 |              52 |           162 |              55 |           184 |
| loso     |    2 | hbo  | gnn            | native    |              0.060 |          0.111 |           0.429 |              0.443 |              49 |            88 |              52 |            98 |
| loso     |    2 | hbr  | gnn            | native    |              0.066 |          0.111 |           0.429 |              0.472 |              49 |            88 |              55 |           102 |
| loso     |    2 | hbt  | gnn            | native    |              0.112 |          0.111 |           0.538 |              0.446 |              49 |            88 |              51 |            97 |
| loso     |    4 | hbo  | gnn            | native    |             -0.166 |          0.000 |           0.250 |              0.425 |              47 |           163 |              54 |           191 |
| loso     |    4 | hbr  | gnn            | native    |             -0.241 |          0.000 |           0.250 |              0.497 |              47 |           163 |              59 |           202 |
| loso     |    4 | hbt  | gnn            | native    |             -0.215 |          0.000 |           0.250 |              0.489 |              47 |           163 |              57 |           187 |