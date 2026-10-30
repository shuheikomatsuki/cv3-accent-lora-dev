# 手順6 動作確認（2026-10-30、helen）

研究計画書 §18.1 手順6。学習設定は §11 の初期候補とUtterTune `configs/train/jsutjvs.yaml` に合わせた仮の値で、決定ではない。
合格基準は実行前に決めたもの。数値の出典は同じフォルダのファイル。

## 設定（仮）

- 学習発話：学習セット（除外後11,716発話）からseed 20261027で300発話（`train_utts.txt`、全発話タグ付き）。JVSは前後カット後のtoken。
- LoRA：Q/K/V/O、r=16・alpha=64（scaling 4.0）・dropout 0.05。`peft.inject_adapter_in_model`で`lm.llm.model`にその場で注入（`src/cv3lora/lora.py`）。
- タグ2行：埋め込み全体は凍結し、151924・151925の位置だけを別の2行パラメータで差し替える。Adamは要素ごとの更新なので、既存行の勾配マスク＋埋め込みのweight decay 0と同じ。初期値は既存の2行（両行とも同じノルム0.2917、既存語彙行の平均0.535）。
- 最適化：AdamW（0.9/0.98、eps 1e-8、weight decay 0）、lr 1e-4、warmup 15ステップ（5%）＋cosine、勾配ノルムのクリップ1.0、バッチ8×300ステップ（約8エポック）、基盤fp32＋bf16 autocast。
- 実行：`scripts/train/01_pilot_train.py` → `scripts/infer/01_gen_tokens.py` → `scripts/infer/02_token2wav.py` → `scripts/eval/01_accent_pairs.py`。図は`scripts/train/02_plot_log.py`。

## 結果

| # | 基準 | 結果 | 根拠 |
|---|---|---|---|
| 1 | タグが各1トークン・接頭辞のIDが変わらない | 合格 | `train.json` checks.tag_tokens（151924・151925、接頭辞一致） |
| 1 | lossが全ステップ有限、最後20ステップ平均 < 最初20ステップ平均 | 合格 | 非有限0回。3.833 → 3.244（`train.json` checks.loss） |
| 1 | 学習300発話のteacher forcing lossが学習前より低い | 合格 | 3.921 → 3.261（`dev_log.tsv`） |
| 1 | 開発20文（タグ付き、seed 0）がすべてEOSで終わり、長さがJSUT録音の0.5〜2倍 | 合格 | 20/20がEOS、比0.83〜1.16（`tokens_dev20_pilot.tsv`） |
| 2 | タグ2行が変わる | 合格 | 初期値からのL2距離0.052・0.056（`train.json` checks.weights） |
| 2 | 他の151,934行と、LoRA・タグ2行以外の全パラメータが学習前と完全に同じ | 合格 | 実効埋め込みの差の最大0.0、変化した凍結パラメータ0/292 |
| 2 | 保存→読み直しのlogitsがメモリ上と一致（差1e-5未満） | 合格 | 差の最大0.0（開発16発話） |
| 3 | 開発10文のtokenを1回生成し、8参照（JA 4・EN 4）で80本 | 合格 | 80本、全wavで読み込んだtokenのsha256が生成時と一致（`wav_dev20_pilot.json`） |
| 3 | 長さ = token数÷25秒±0.1秒 | 合格 | 誤差の最大0.000秒 |
| 3 | 同じseedで再実行すると同じ出力（差1e-4未満） | 合格 | 差の最大0.0（flowのCFMノイズは固定buffer、HiFTの乱数はseed固定） |
| 3 | 参照を変えると出力が変わる | 合格 | 同じtokenで参照どうしの平均絶対差の最小0.021 |
| 4 | A（基盤・端橋箸鼻花）とC（試行LoRA・ハシ ハシ' ハ'シ ハナ ハナ'）を3 seed×JA開発4名で合成し、ダでのF0変化の計測まで最後まで動く | 合格 | 120本すべて整列が通った。ダでの変化が出せたのはA 50/60・C 30/60（残りは語末モーラのF0が無声で欠測）（`accent_pairs.tsv`・`accent_pairs_summary.tsv`） |

記録の仕組み：`train_log.tsv`（毎ステップ：loss・token数・勾配ノルム・学習率・非有限の回数・タグ2行のノルムと初期値からの距離・LoRAのノルム・秒）、`dev_log.tsv`（50ステップごと：開発249発話と学習300発話のteacher forcing loss・正解率。ステップ0は基盤モデルと同じ）、図`train_log.png`。動いた。

## 気づいた点（判断はしていない）

1. 基盤モデルは短いキャリア文で出力が崩れる。「これは橋だと思います。」はseed 0で0.48秒。Aは0.48〜1.72秒（中央値1.08秒）で、試行LoRAのC（1.48〜3.04秒、中央値1.72秒）より短い（`wav_pairs_*.tsv`）。jvs029の15本をCTCで貪欲復号すると、Aでは「はし」「はな」がどれにも出てこず、Cでは15本すべてに出た（scratchの確認スクリプトなので結果ファイルはない）。公式の`inference_cross_lingual`でも同じseedで同じ長さになるので、自前の生成コードのせいではない。原因と思われるもの：`sampling_ids`が最小長までの間に止めるのは`speech_token_size`（sos）で、EOS（`speech_token_size+1`）は止めないので、`min_token_text_ratio`が効かない（`third_party/CosyVoice/cosyvoice/llm/llm.py` の `sampling_ids` と `CosyVoice3LM.__init__`）。ただし後の調査で、この修正を当てても内容は直らなかった（`results/short_text/summary.md`）。§13.4の条件Aで既定の型を記録するには、このままでは使えない。Aの「既定の型」（`accent_pairs_summary.tsv`）は崩れた音声での値なので、記録として扱わない。
2. 強制整列は、読みが違っても位置を返す。F0の値を使う前に、読みが合っているかを確かめる手順（ASRや整列のスコア）が要る。
3. 2モーラ語「ハシ」の語末「シ」は無声になりやすく、F0が欠測しやすい（C_ハシは12本中3本しか計測できなかった）。
4. 勾配ノルム（クリップ前）は全ステップで1.4〜6.5、つまりクリップ1.0は毎ステップかかっている（UtterTuneと同じ値）。
5. 開発lossは4.268 → 3.803で、150ステップ以降はほぼ横ばい。ピークメモリ7.05GB、1ステップ0.152秒（バッチ8、評価を除く）。
6. ハシ・ハナは§13.4の例の語。最終の対照テストに入れるなら、この試行の結果を設定選びに使わないこと。
