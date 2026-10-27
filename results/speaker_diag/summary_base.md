# Speaker-information loss diagnostic: base

value = NLL per speech token; split seed 20261027

## Set A

- JVS speakers 100, utterances 1161 (per speaker 6–19)
- train speakers: mean 3.5233, SD 0.2258 (nats per speech token)
- split-half reliability of the per-speaker value (Spearman–Brown): 0.87

| group | n | mean | mean z vs train |
|---|---:|---:|---:|
| train | 76 | 3.5233 | -0.00 |
| dev | 4 | 3.5697 | +0.21 |
| eval | 20 | 3.4829 | -0.18 |

| dev speaker | gender | utts | value | z vs train | percentile in train |
|---|---|---:|---:|---:|---:|
| jvs029 | F | 7 | 3.6140 | +0.40 | 68 |
| jvs034 | M | 11 | 3.8590 | +1.49 | 97 |
| jvs053 | F | 10 | 3.4666 | -0.25 | 33 |
| jvs086 | M | 12 | 3.3392 | -0.82 | 12 |

## Set B

- JVS speakers 100, utterances 500 (per speaker 5–5)
- train speakers: mean 3.4643, SD 0.2459 (nats per speech token)
- split-half reliability of the per-speaker value (Spearman–Brown): 0.93

| group | n | mean | mean z vs train |
|---|---:|---:|---:|
| train | 76 | 3.4643 | +0.00 |
| dev | 4 | 3.4913 | +0.11 |
| eval | 20 | 3.3578 | -0.43 |
| jsut | 1 | 3.8779 | +1.68 |

| dev speaker | gender | utts | value | z vs train | percentile in train |
|---|---|---:|---:|---:|---:|
| jvs029 | F | 5 | 3.4221 | -0.17 | 36 |
| jvs034 | M | 5 | 3.6412 | +0.72 | 78 |
| jvs053 | F | 5 | 3.7155 | +1.02 | 89 |
| jvs086 | M | 5 | 3.1862 | -1.13 | 11 |
