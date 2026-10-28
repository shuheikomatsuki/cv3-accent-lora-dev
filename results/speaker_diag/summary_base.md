# Speaker-information loss diagnostic: base

value = NLL per speech token; split seed 20261027

## Set A

- JVS speakers 100, utterances 1160 (per speaker 6–19)
- train speakers: mean 3.9840, SD 0.2177 (nats per speech token)
- split-half reliability of the per-speaker value (Spearman–Brown): 0.74

| group | n | mean | mean z vs train |
|---|---:|---:|---:|
| train | 76 | 3.9840 | +0.00 |
| dev | 4 | 4.0319 | +0.22 |
| eval | 20 | 3.9687 | -0.07 |

| dev speaker | gender | utts | value | z vs train | percentile in train |
|---|---|---:|---:|---:|---:|
| jvs029 | F | 7 | 4.1141 | +0.60 | 75 |
| jvs034 | M | 11 | 4.2573 | +1.26 | 88 |
| jvs053 | F | 10 | 3.8690 | -0.53 | 28 |
| jvs086 | M | 12 | 3.8873 | -0.44 | 29 |

## Set B

- JVS speakers 100, utterances 500 (per speaker 5–5)
- train speakers: mean 3.7669, SD 0.2485 (nats per speech token)
- split-half reliability of the per-speaker value (Spearman–Brown): 0.92

| group | n | mean | mean z vs train |
|---|---:|---:|---:|
| train | 76 | 3.7669 | -0.00 |
| dev | 4 | 3.7580 | -0.04 |
| eval | 20 | 3.6546 | -0.45 |
| jsut | 1 | 3.8779 | +0.45 |

| dev speaker | gender | utts | value | z vs train | percentile in train |
|---|---|---:|---:|---:|---:|
| jvs029 | F | 5 | 3.6791 | -0.35 | 33 |
| jvs034 | M | 5 | 3.9121 | +0.58 | 74 |
| jvs053 | F | 5 | 4.0481 | +1.13 | 88 |
| jvs086 | M | 5 | 3.3929 | -1.51 | 5 |
