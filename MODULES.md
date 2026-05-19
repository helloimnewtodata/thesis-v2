# MODULES.md

Käyttöohje ja muistilappu repon `.py`-tiedostoista. Mitä kukin tekee, mitä funktioita sisältä löytyy ja miten ne pelaavat yhteen pipelinessa.

## Pipeline — marssijärjestys

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 0 — Universumi (kertaluonteinen)                                     │
│   python -m src.build_survivor_stoxx_universe                              │
│       →  data/survivor_universe.csv                                        │
└────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Data Fetch (tarvittaessa)                                        │
│   python scripts/fetch_ecb_cache.py    →  ECB AAA/all curve cache          │
│   src/data_fetch.py                    →  Refinitiv: prices, fundamentals, │
│                                           quarterly EPS/CF, index, EURIBOR │
└────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Regiimimallit (erilliset ajot, eivät riipu featureista)          │
│   python -m src.HMM    →  data/01_raw/outputs/                             │
│                           hmm_regimes_monthly_no_lookahead_ml.csv          │
│   python -m src.JM2    →  JM2_output_ml.csv  (vaihtoehtoinen, soft JM)     │
└────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Master-paneeli                                                   │
│   scripts/updated_main_test.py  (orkestroi: data_fetch → features →        │
│                                  hurst → HMM merge)                        │
│       käyttää:  src/pipeline.py, src/data_fetch.py, src/features.py,       │
│                 src/hurst.py                                               │
│   →  data/02_preprocessed/MASTER_DF_1.csv                                  │
└────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 3b — Point-in-time master + tuotantopaneeli                          │
│   scripts/PIT_universe.py               →  PIT_MASTER_DF_1.csv             │
│   scripts/merge_jm2_to_master.py        →  PIT_MASTER_DF_1_JM2.csv         │
│   scripts/build_production_master.py    →  MASTER_DF_PROD_JM2_nonnan_      │
│       --winsorize-stock-continuous          winsor.csv                     │
└────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 4 — Mallit + portfolio (walk-forward OOS)                            │
│   python scripts/run_prediction.py                                         │
│       →  results/predictions_oos_*.csv                                     │
│       →  results/evaluation_metrics_*.csv                                  │
│       →  results/portfolio_returns_*.csv                                   │
│       →  results/portfolio_performance_*.csv                               │
└────────────────────────────────────────────────────────────────────────────┘
```

End-to-end ajo (kun universumi, ECB-cache ja regiimit ovat olemassa):
```bash
python scripts/updated_main_test.py
python scripts/PIT_universe.py
python scripts/merge_jm2_to_master.py --master data/02_preprocessed/PIT_MASTER_DF_1.csv
python scripts/build_production_master.py --winsorize-stock-continuous
python scripts/run_prediction.py
```

> **Asennus**: `pip install -e .` repon juuressa tekee `src/`, `scripts/`, `diagnostics/`, `experiments/` -hakemistoista importoitavia paketteja. Tämä on edellytys sille että yllä olevat komennot toimivat.

---

## Top-level

### `config.py`
Jaetut vakiot. Päivämäärät (`FETCH_START_DATE`, `DISPLAY_START_DATE`), Refinitiv-hakujen `CHUNK_SIZE` / `SLEEP_BETWEEN_CHUNKS`, sektoridummien määritelmät (`SECTOR_DUMMIES`, `SECTOR_DUMMY_NAMES`) ja train/val/OOS-aikaikkunoiden rajat. Importataan kaikkialta top-level py-modulena. Ei funktioita.

### `pyproject.toml`
Pakettimetadata + riippuvuudet (lähde). `requirements.txt` on synkattu peilikopio.

---

## `src/` — kirjastokoodi

Importoitava `from src.X import Y`. `src/` ei ole entry-pointtien koti; ajettavat skriptit ovat `scripts/`-, `diagnostics/`- ja `experiments/`-kansioissa.

### `src/pipeline.py`
Kirjastomoduuli master-paneelin rakennukseen. `scripts/updated_main_test.py`, `scripts/PIT_universe.py` ja `diagnostics/point_in_time_smoke_test.py` käyttävät tätä. (Aiemmin `main.py` repon juuressa; siirretty 2026-05.)

| Funktio | Tehtävä |
|---|---|
| `load_universe(path)` | Lukee universumi-CSV:n RIC-listaksi. |
| `_ensure_date_column(df)` | Varmistaa että DataFramessa on `Date`-sarake datetime-tyyppisenä. |
| `to_monthly_stock_panel(df_features)` | Resamplaa per-stock päivädatan kuukausilopuiksi. |
| `to_monthly_index_features(df_idx)` | Sama indeksifeatureille. |
| `build_hurst_panel(df_features)` | Laskee `hurst.compute_hurst_dfa`-kutsun ja palauttaa kuukausittaisen Hurst-paneelin. |
| `load_hmm_panel(path)` | Lukee HMM-CSV:n (regiimit + posteriorit). |
| `build_master_dataframe(df_features, df_idx, hmm_path)` | Yhdistää stock + index + Hurst + HMM yhdeksi master-DataFrameksi. |

Vakiot: `HMM_ML_PATH`, `HMM_PROBABILITY_COLUMNS`.

### `src/build_survivor_stoxx_universe.py`
Rakentaa STOXX Europe 600 -survivor-bias-korjattua universumia. Kertaluonteinen ajo (output cachettuu CSV:hen).

| Funktio | Tehtävä |
|---|---|
| `_fetch_snapshot_rics(snapshot_date)` | Hakee indeksin jäsenet annetun päivämäärän snapshotissa. |
| `fetch_index_change_events(start, end, index_ric)` | Hakee kaikki indeksiin/poistumis-tapahtumat aikavälillä. |
| `filter_continuous_members(start_snapshot, events, start, end)` | Suodattaa ne RIC:t, jotka olivat indeksissä koko aikavälin. |
| `filter_by_data_availability(rics, check_date)` | Pudottaa RIC:t, joilla ei ole hintadataa `check_date`:hen mennessä (252-päivän warmup-vaatimus). |
| `fetch_survivor_universe(start, end)` | Yhdistää yllä olevat → survivor-lista + diagnostiikka. |
| `save_survivor_universe(...)` | Tallentaa CSV:n (`data/survivor_universe.csv`). |
| `main()` | CLI-entrypoint. |

### `src/build_memberships.py`
Rakentaa STOXX Europe 600 -jäsenyysintervallit (point-in-time universumin pohja). Käyttää `build_survivor_stoxx_universe.py`:n apufunktioita.

### `src/data_fetch.py`
Refinitiv LSEG -kutsut. Chunkattu, retry-logiikka, rekursiivinen halving epäonnistuneille chunkeille.

| Funktio | Tehtävä |
|---|---|
| `open_session()` | Avaa Refinitiv-session (käyttää lokaalia Workspacea). |
| `_fetch_in_chunks(...)` | Geneerinen chunk-fetcher: yrittää, halvaa chunk kahtia jos epäonnistuu, yhdistää tulokset. |
| `_clean_stock_fundamentals(df)` | Poistaa duplikaatit, normalisoi sarakkeet. |
| `_coerce_numeric_columns(df, exclude)` | Pakottaa numeeriset sarakkeet floatiksi. |
| `fetch_stock_fundamentals(universe)` | Hakee päivätason fundamentaalit (P/B, P/S, P/CF, EPS, market cap, ...). |
| `fetch_quarterly_eps_for_pe(universe)` | Hakee kvartaalitason EPS:n (käytetään `E/P_ff`:n fallback-ketjussa). |
| `fetch_quarterly_pcf_for_pcf(universe)` | Hakee kvartaalitason operating cash flow + per-share (käytetään `-P/CF_ff`:n fallback-ketjussa). |
| `fetch_stock_prices(universe)` | Päivätason hintadata. |
| `fetch_index_data()` | STOXX Europe 600 -indeksin hinnat ja fundamentaalit. |
| `trim_warmup(df, start_date)` | Pudottaa warmup-jakson features-rakennuksen jälkeen. |
| `fetch_euribor()` | Hakee 3kk EURIBOR-koron. |

### `src/ecb_cache_utils.py`
ECB-välimuistin jaettu lataaja. Käytetään regiimimalleista (`HMM*`, `JM*`, `CJM*`, `GMM`, `MSR`, `hmm_msr_benchmark`). Palauttaa `None` jos cache-ikkuna on >7 päivää lyhempi kuin pyydetty alue (jolloin ladataan uudelleen).

| Funktio | Tehtävä |
|---|---|
| `build_ecb_urls(start, end)` | Rakentaa ECB SDW -URL:t pyydetylle aikavälille. |
| `load_cached_ecb_series(...)` | Yrittää lukea cachen — palauttaa `None` jos riittämätön. |
| `download_bytes_with_urlopen(url, ssl_verify)` | HTTP-haku stdlibillä. |
| `download_bytes_with_curl(url, ssl_verify)` | Fallback `curl`-subprocessilla. |
| `download_csv_bytes(url, ssl_verify)` | Yhdistää urlopen + curl-fallback. |
| `download_and_cache_ecb_series(url, cache_path, ...)` | Lataa + tallentaa cacheen. |
| `get_ecb_series(...)` | Pääentry: cache-first, lataa tarvittaessa. |

### `src/features.py`
Per-stock feature-rakennus. Sisältää valuaatio-, laatu-, momentum-, riski- ja standalone-featuret.

| Funktio | Tehtävä |
|---|---|
| **Valuaatio** | |
| `compute_ep(df)` | E/P (perusversio Refinitivin fundamenteista). |
| `compute_pe_ltm(df)` | LTM P/E suoraan Refinitivistä. |
| `compute_pe_ttm_fallback(df, df_quarterly_eps)` | **Tuotanto-E/P**: 3-vaiheinen fallback (LTM EPSfr → 4Q TTM EPSfr → NI/Shares) → `E/P_ff`, `P/E_ff`. |
| `compute_inv_pb(df)` | `1/(P/B)` (orientaatio: korkeampi = halvempi). |
| `compute_neg_ps(df)` | `-(P/S)`. |
| `compute_pcf_op(df)` | `-P/CF_op_LTM` ja `-P/CF_opps_LTM` LTM-operating-CF:stä. |
| `compute_pcf_ttm_fallback(df, df_quarterly_pcf)` | **Tuotanto-P/CF**: fallback (LTM op CF → 4Q TTM → per-share) → `-P/CF_ff`. |
| `compute_dividend_yield_trailing(df)` | Trailing 12kk osingot / edellinen kuukausilopun hinta (`merge_asof`). |
| **Laatu** | |
| `compute_neg_debt_to_mktcap(df)` | `-(Total Debt / Market Cap)`. |
| `compute_book_to_market(df)` | Common Equity / Market Cap. (Lasketaan välipaneeliin, mutta `scripts/build_production_master.py` jättää sen oletuksena pois tuotantopaneelista redundanssin takia `1/P/B`:n kanssa.) |
| `compute_operating_profitability(df)` | EBITDA / lagged Common Equity (Fama-French 2015). |
| **Momentum** | |
| `compute_mom_1m(df)` | 1kk total return. |
| `compute_mom_12m(df)` | 12kk return, jättää kuukauden t−1 yli (Jegadeesh & Titman 1993). |
| `compute_sector_group(df)` | Liittää GICS-sektoriryhmät. |
| `compute_stock_vs_sector_return(df)` | Leave-one-out stock-vs-sector return (1M ja 12M-1M, Moskowitz & Grinblatt 1999). |
| `compute_rsi(df, window=30)` | RSI-indikaattori. |
| `compute_daily_return(df)` | Päiväreturn per stock. |
| **Riski** | |
| `compute_volatility(df, window=30)` | 30 päivän rolling stdev. |
| `compute_beta(df, df_index)` | 252-päivän rolling beta vs. .STOXXR. |
| `compute_idiosyncratic_vol(df, df_index, window=252)` | CAPM-residuaalin stdev. |
| **Indeksi** | |
| `compute_index_features(df_index_fundamentals)` | Indeksin valuaatio-/momentum-featuret. |
| **Standalone** | |
| `compute_log_market_cap(df)` | `log(Market Cap)`. |
| `compute_sector_dummies(df, sectors, names)` | 5 binääristä Sector_Group-dummia GICS 11 → 6 -aggregaation pohjalta: `Sector_Financials`, `Sector_Industrials_Materials`, `Sector_Consumer`, `Sector_Health_Care`, `Sector_Technology_Communication`. Referenssikategoria: Real Assets & Utilities (Energy + Utilities + Real Estate). `sectors`/`names`-parametrit hyväksytään yhteensopivuuden takia mutta jätetään huomiotta — dummit luetaan aina `_SECTOR_DUMMY_GROUPS`-listasta. |
| `compute_excess_return(df, df_euribor)` | Stock return − 3kk EURIBOR (kuukausitettu). |
| **Orkestrointi** | |
| `compute_all_features(...)` | Kutsuu kaikki yllä olevat oikeassa järjestyksessä → täysi feature-paneeli. |

### `src/hurst.py`
Hurst-eksponentti DFA-menetelmällä, rinnakkaistettu per stock.

| Funktio | Tehtävä |
|---|---|
| `_compute_hurst_dfa(returns, min_obs)` | Yhden stockin yhden ikkunan DFA-Hurst (`antropy.detrended_fluctuation`). |
| `_rank_normalize(feature_df)` | Cross-sectional rank → arvo välille [−1, 1]. |
| `_process_stock(stock, wide_returns, month_ends, window, min_obs)` | Kuukausilopuissa rolling-Hurst per stock. |
| `compute_hurst_dfa(daily_df, window=252, min_obs=252, n_jobs=-1)` | Pääentry: rinnakkaistettu Hurst koko paneelille → kaksi saraketta, `Hurst_Raw_DFA` (raakaarvo `(0, 1)`) ja `Hurst` (cross-sectional rank `[−1, +1]`). Tuotantopaneelista jätetään oletuksena raw pois — vain ranked päätyy `MASTER_DF_PROD_JM2_*.csv`:hen. |

### `src/HMM.py`
No-lookahead Gaussian HMM -regiimiluokittelija (Bull / Bear / Transition). Tuotantokäytössä. Output: `data/01_raw/outputs/hmm_regimes_monthly_no_lookahead_ml.csv`. Ajetaan `python -m src.HMM`.

| Funktio | Tehtävä |
|---|---|
| `fetch_input_data(start, end)` | STOXX-hinnat + ECB-tuottokäyrät (AAA, all). |
| `build_feature_matrix(df_equity, df_yields, df_credit)` | Rakentaa HMM-input-featuret (return, volatility, term spread, credit spread). |
| `get_month_end_observation_dates(hmm_df)` | Listaa kuukausilopun observointipäivät. |
| `make_regime_map(raw_states, train_df)` | Mappaa raaka HMM-statet (0,1,2) ekonomiseen järjestykseen → IDt stabiileja refittien yli. |
| `fit_predict_month_end_regime(train_df, features)` | Fittää HMM:n trainiin, ennustaa kuukausilopun regiimin. |
| `select_train_window(hmm_df, source_date)` | Train-ikkuna alusta `source_date`:hen (expanding window, ei lookaheadia). |
| `build_no_lookahead_monthly_regimes(hmm_df, features)` | Pää: per kuukausi refit + ennuste → koko aikasarja regiimejä. |
| `build_ml_safe_monthly_regimes(monthly)` | Lopullinen ML-turvallinen versio (regiimit lagattu, jotta saatavissa kuukausilopussa t). |
| `main()` | CLI-entrypoint. |

### `src/HMM2.py`, `src/HMM3.py`
Timing-Sharpe-optimoidut HMM-variantit (HMM.py:n johdannaiset). HMM2: K=2, multi-restart, diag cov, feature smoothing. HMM3: lisää sticky transition + realized vol + EWMA posterior smoothing. Vertailukäytössä `experiments/regime_benchmarks/`-skripteissä. Output: `hmm2_/hmm3_regimes_monthly_no_lookahead_ml.csv`.

### `src/JM.py`
Hard discrete Statistical Jump Model (Bemporad et al. 2018; Nystrup et al. 2020). Vertailukäytössä; tuottaa `JM_output_ml.csv`.

### `src/JM2.py`
Naiivi continuous JM -yritys (tavallinen L1) — paperin mukaan romahtaa hard JM:ksi. Säilytetty referenssinä; tuottaa `JM2_output_ml.csv` (käytetään `merge_jm2_to_master.py`:n input-tiedostona, jos halutaan JM2-regiimien todennäköisyydet masteriin).

### `src/CJM.py`
Continuous Statistical Jump Model paperin mukaisesti (Aydinhan et al. 2024) — neliöity L1, hilaratkaisu. Tuottaa `CJM_output_ml.csv`.

### `src/CJM2.py`
Sama tavoitefunktio kuin `CJM.py`, mutta state-osaongelma ratkaistaan aitona jatkuvana QP:na FISTA:lla (ei hilaa). Tuottaa `CJM2_output_ml.csv`.

### `src/GMM.py`
Gaussian Mixture Model -pohjainen regiimiluokittelu. Vaihtoehto HMM:lle.

### `src/MSR.py`
Markov-Switching Regression -malli (statsmodels). Vertailukäytössä.

### `src/hmm_msr_benchmark.py`
HMM vs. Markov-switching -vertailu. Output: `hmm_vs_markov_switching_benchmark.csv` + `crosstab` / `summary`.

### `src/sentiment.py`
FinBERT-pohjainen uutissentimentin skoraus per stock-month (sentiment pipelinen vaihe 2). Input: `data/01_raw/news_headlines_raw.csv` (tuotettu `src/fetch_news.py`:llä). Output: `data/02_preprocessed/MASTER_DF_PROD_JM2_SENTIMENT.csv`.

### `src/fetch_news.py`
Refinitiv-uutisotsikoiden haku STOXX 600 -universumille kuukausi-RIC-tasolla (sentiment pipelinen vaihe 1).

---

## `scripts/` — CLI entry pointit

Ajetaan repon juuresta `python scripts/<name>.py`.

### `scripts/updated_main_test.py`
**Päämaster-ajo.** Hakee Refinitivistä uudelleen, rakentaa featuret ja kirjoittaa `MASTER_DF_1.csv`:n. Käyttää `src/pipeline.py`:n `build_master_dataframe`:a.

| Funktio | Tehtävä |
|---|---|
| `default_fetch_start(display_start)` | Päättelee fetch-aloituspäivän (warmup) display-startista. |
| `parse_args()` | CLI-argumentit (`--end-date`, `--chunk-size`, ...). |
| `apply_runtime_overrides(...)` | Päivittää `config`- ja `data_fetch`-moduulien runtime-vakiot CLI:n perusteella. |
| `drop_extra_valuation_columns(df)` | Pudottaa ei-tuotannolliset valuaation duplikaatit (esim. `P/E_ff`, `-P/CF_op_LTM`). |
| `reorder_master_columns(df, order)` | Asettaa sarakejärjestyksen `MASTER_COLUMN_ORDER`-listan mukaiseksi. |
| `save_outputs(prefix, output_dir, datasets)` | Tallentaa raakatulokset `data/01_raw/`-hakemistoon. |
| `main()` | Pääajo: fetch → feature build → master → CSV. |

### `scripts/PIT_universe.py`
Point-in-time universumin rakentaja. Tuottaa `PIT_MASTER_DF_1.csv`:n joka sisältää PIT-eligibility-flagit (`IsIndexMemberAtT`, `HasCoreHistory`, `HasNextMonthReturn`).

### `scripts/build_production_master.py`
Tuotantomasterin rakentaja PIT-masterista: suodattaa PIT-flagien mukaan, laskee monthly excess return ja next-month target, kääntää indeksin P/E ja P/B → E/P ja B/M. Optio `--winsorize-stock-continuous` klippaa continuous featuret 1%/99%-rajoille per kuukausi.

### `scripts/merge_jm_to_master.py`
Mergeaa hard JM -regiimi-indikaattorit masteriin (`MASTER_DF_1.csv` + `JM_output_ml.csv` → `MASTER_DF_1_JM.csv`).

### `scripts/merge_jm2_to_master.py`
Mergeaa JM2-regiimi-todennäköisyydet masteriin. Tukee `--master`-argumenttia (esim. PIT-master).

### `scripts/merge_cjm_to_master.py`
Mergeaa CJM-regiimi-todennäköisyydet masteriin.

### `scripts/run_prediction.py`
**Cross-sectional excess-return prediction pipeline** (korvaa aiemman `src/models.py` + `src/portfolio.py`-yhdistelmän — ne ovat arkistoituna `archive/legacy_pipeline/`-kansiossa).

Tukee linja-malleja (OLS, Ridge, LASSO), gradient boostingia (LightGBM, XGBoost), feedforward-NN:ää (3-kerroksinen, IC loss) ja näiden ensembliä. Kaksivaiheinen walk-forward (Phase 1: kiinteät train+val Hyperparametrien valintaan; Phase 2: expanding-window kuukausittainen refit OOS-ennusteille). Sisältää myös portfolion rakennuksen ja arviointitestit (Clark-West, Diebold-Mariano).

Input: `data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv`.
Outputs: `results/predictions_oos.csv`, `ic_timeseries.csv`, `portfolio_returns.csv`, `metrics.csv`, `portfolio_performance.csv`, `clark_west.csv`, `diebold_mariano.csv`, `decile_returns.csv`, `decile_monotonicity.csv`, `turnover.csv`.

### `scripts/run_sentiment_experiment.py`
Sentiment pipelinen vaihe 3: kolmen mallivariantin (A: ei sentimentiä, B: news_dummy, C: täysi sentimentti) vertailu OOS-ikkunassa. Pooled OLS + Newey-West HAC -regressio kertoo, parantaako sentimentti suorituskykyä yli pelkän uutiskattavuuden.

### `scripts/fetch_ecb_cache.py`
ECB-tuottokäyrän kertaluonteinen lataaja (AAA + all curve, 10y). Output: `data/01_raw/outputs/ecb_cache/`. Kutsuu `src/ecb_cache_utils.py`:n `get_ecb_series`-funktiota.

---

## `diagnostics/` — kertaluonteiset diagnostiikat

### `diagnostics/beta_idiosync_diagnostics.py`
Beta_252d ja -IdioVol NaN-osuuksien tarkistus warmup-kalibrointia varten.

### `diagnostics/check_stoxxr.py`
Tarkistaa `.STOXXR`-indeksin saatavuuden ja peruslaadun.

### `diagnostics/point_in_time_smoke_test.py`
Pieni point-in-time smoke-testi: samplaa muutaman RIC:n, ajaa pipelinen warmup-datalla ja kirjoittaa eligibility-flagit. Käytä `scripts/PIT_universe.py`:n sijaan kun haluat nopean sanity-tarkistuksen.

---

## `experiments/` — tutkimusajot

### `experiments/xgb_walkforward_hpo.py`
XGBoost-hyperparametrien grid- tai random-search walk-forward-asetelmassa. Output: `results/xgb_walkforward_hpo_best_params.json` + `xgb_walkforward_hpo_results.csv`.

### `experiments/regime_benchmarks/regime_strategy_benchmark_all.py`
Vertaa kaikkia regiimimalleja (HMM, HMM2, HMM3, GMM, JM, JM2, CJM) timing-strategiassa "1 − P(Bear)" -painotuksella. Output: `regime_strategy_metrics_all.csv` + plotit.

### `experiments/regime_benchmarks/regime_strategy_benchmark_shu_01.py`
Sama vertailu Shu et al. -paperin (2024) 0/1-allokaatiolla.

---

## `archive/` — historiakerros (ei käytössä)

### `archive/legacy_pipeline/`
`models.py` ja `portfolio.py` repon entisestä `src/`-kansiosta. Korvattu `scripts/run_prediction.py`:llä. Säilytetty referenssinä — ei importattu mistään ajantasaisesta koodista.

### `archive/regime_iterations/`
`regime_strategy_benchmark.py` ja sen `_soft`/`_soft2`/`_soft3`-iteraatiot. Korvattu `_all.py`:llä ja `_shu_01.py`:llä `experiments/regime_benchmarks/`-kansiossa.

### `archive/data/`, `archive/results/`
Vanhentuneet master-CSV:t ja superseded results-tiedostot. Gitignored.

---

## Konventiot
- **`src/`** = importoitava kirjasto. **`scripts/`** = ajettavat CLI-entry-pointit.
- **Ei look-aheadia**: kaikki featuret laskettu vain hetken t mennessä saatavilla olevasta datasta. `merge_asof(direction="backward")` valuaatio-fundamenttien spreadissa päivätasolle.
- **Orientaatio**: kaikki valuaatiofeatures masterissa siten että **korkeampi = halvempi = osta-signaali**.
- **`data/`** ja **`results/`** ovat gitignored — ei näy fresh kloonissa.
- **Asennus**: `pip install -e .` repon juuressa tekee kaikki paketit importoitaviksi.

Katso lisätietoja: `CLAUDE.md` (metodologia), `README.md` (yleiskuva), `Research_Plan_v2.pdf` (akateeminen tausta).
