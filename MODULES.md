# MODULES.md

Käyttöohje ja muistilappu repon `.py`-tiedostoista. Mitä kukin tekee, mitä funktioita sisältä löytyy ja miten ne pelaavat yhteen pipelinessa.

## Pipeline — marssijärjestys

```
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 0 — Universumi (kertaluonteinen)                                │
│   src/build_survivor_stoxx_universe.py  →  data/survivor_universe.csv │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Data Fetch (tarvittaessa)                                   │
│   src/fetch_ecb_cache.py        →  ECB AAA/all curve cache            │
│   src/data_fetch.py             →  Refinitiv: prices, fundamentals,   │
│                                    quarterly EPS/CF, index, EURIBOR   │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 2 — HMM regiimit (erillinen ajo, ei riipu featureista)          │
│   src/HMM.py                    →  data/01_raw/outputs/               │
│                                    hmm_regimes_monthly_no_lookahead_  │
│                                    ml.csv                             │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Master-paneeli                                              │
│   updated_main_test.py  (orkestroi: data_fetch → features →           │
│                           hurst → HMM merge)                          │
│       käyttää:  main.py, src/data_fetch.py, src/features.py,          │
│                 src/hurst.py                                          │
│   →  data/02_preprocessed/MASTER_DF_1.csv                             │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 4 — Mallit (walk-forward OOS)                                   │
│   python -m src.models          →  results/predictions_oos.csv,       │
│                                    results/evaluation_metrics.csv     │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 5 — Portfolio                                                   │
│   python -m src.portfolio       →  results/portfolio_returns.csv,     │
│                                    results/portfolio_performance.csv  │
└───────────────────────────────────────────────────────────────────────┘
```

End-to-end ajo (kun universumi ja HMM-CSV on jo olemassa):
```bash
python updated_main_test.py    # Stage 3
python -m src.models           # Stage 4
python -m src.portfolio        # Stage 5
```

---

## Top-level

### `config.py`
Jaetut vakiot. Päivämäärät (`FETCH_START_DATE`, `DISPLAY_START_DATE`), Refinitiv-hakujen `CHUNK_SIZE` / `SLEEP_BETWEEN_CHUNKS`, sektoridummien määritelmät (`SECTOR_DUMMIES`, `SECTOR_DUMMY_NAMES`) ja train/val/OOS-aikaikkunoiden rajat. Importataan kaikkialta. Ei funktioita.

### `main.py`
Kirjastomainen pipeline-orkestroija. `updated_main_test.py` käyttää tätä master-paneelin koonnissa.

| Funktio | Tehtävä |
|---|---|
| `load_universe(path)` | Lukee universumi-CSV:n RIC-listaksi. |
| `_ensure_date_column(df)` | Varmistaa että DataFramessa on `Date`-sarake datetime-tyyppisenä. |
| `to_monthly_stock_panel(df_features)` | Resamplaa per-stock päivädatan kuukausilopuiksi. |
| `to_monthly_index_features(df_idx)` | Sama indeksifeatureille. |
| `build_hurst_panel(df_features)` | Laskee `hurst.compute_hurst_dfa`-kutsun ja palauttaa kuukausittaisen Hurst-paneelin. |
| `load_hmm_panel(path)` | Lukee HMM-CSV:n (regiimit + posteriorit). |
| `build_master_dataframe(df_features, df_idx, hmm_path)` | Yhdistää stock + index + Hurst + HMM yhdeksi master-DataFrameksi. |
| `main()` | CLI-entrypoint (käytännössä korvattu `updated_main_test.py`:llä). |

### `updated_main_test.py`
**Päämaster-ajo.** Hakee Refinitivistä uudelleen, rakentaa featuret ja kirjoittaa `MASTER_DF_1.csv`:n.

| Funktio | Tehtävä |
|---|---|
| `default_fetch_start(display_start)` | Päättelee fetch-aloituspäivän (warmup) display-startista. |
| `parse_args()` | CLI-argumentit (`--end-date`, `--chunk-size`, ...). |
| `apply_runtime_overrides(...)` | Päivittää `config`- ja `data_fetch`-moduulien runtime-vakiot CLI:n perusteella. |
| `print_summary(name, df)` | Diagnostiikkatulostus kustakin välivaiheen DataFramesta. |
| `prepare_export_df(name, df)` | Siistii export-muotoon (poistaa epätoivotut sarakkeet). |
| `save_outputs(prefix, output_dir, datasets)` | Tallentaa raakatulokset `data/01_raw/`-hakemistoon. |
| `load_universe(path)` | Lukee survivor-universumin. |
| `pick_random_universe(snapshot, size, seed)` | Smoke-testiajoja varten — poimii satunnaisen alijoukon. |
| `nan_report(df, feature_cols)` | Per-sarake NaN-osuudet. |
| `drop_extra_valuation_columns(df)` | Pudottaa ei-tuotannolliset valuaation duplikaatit (esim. `P/E_ff`, `-P/CF_op_LTM`). |
| `reorder_master_columns(df, order)` | Asettaa sarakejärjestyksen `MASTER_COLUMN_ORDER`-listan mukaiseksi. |
| `print_nan_report(df, feature_cols, title)` | Tulostaa NaN-raportin loggeihin. |
| `main()` | Pääajo: fetch → feature build → master → CSV. |

---

## `src/`

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

### `src/fetch_ecb_cache.py`
ECB-tuottokäyrän kertaluonteinen lataaja (AAA + all curve, 10y). Output: `data/01_raw/outputs/ecb_cache/`.

| Funktio | Tehtävä |
|---|---|
| `main()` | Hakee molemmat sarjat ja tallentaa CSV:nä. |

### `src/ecb_cache_utils.py`
ECB-välimuistin jaettu lataaja. Käytetään `HMM.py`:stä. Palauttaa `None` jos cache-ikkuna on >7 päivää lyhempi kuin pyydetty alue (jolloin ladataan uudelleen).

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
| `compute_neg_pcf(df)` | Legacy `-P/CF` (`TR.F.CF` total CF). Tuotannon ulkopuolella. |
| `compute_pcf_op(df)` | `-P/CF_op_LTM` ja `-P/CF_opps_LTM` LTM-operating-CF:stä (ei fallbackia). |
| `compute_pcf_ttm_fallback(df, df_quarterly_pcf)` | **Tuotanto-P/CF**: fallback (LTM op CF → 4Q TTM → per-share) → `-P/CF_ff`. |
| `compute_pcf_refinitiv(df)` | Refinitivin oma `Price To Cash Flow Per Share`. Vertailu/diagnostiikka. |
| `compute_dividend_yield_trailing(df)` | Trailing 12kk osingot / edellinen kuukausilopun hinta (`merge_asof`). |
| **Laatu** | |
| `compute_neg_debt_to_mktcap(df)` | `-(Total Debt / Market Cap)`. |
| `compute_book_to_market(df)` | Common Equity / Market Cap. |
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
| `compute_sector_dummies(df, sectors, names)` | One-hot-sektoridummiet (Cnsmr, Manuf, HiTec, Hlth). |
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
| `compute_hurst_dfa(daily_df, window=252, min_obs=252, n_jobs=-1)` | Pääentry: rinnakkaistettu Hurst koko paneelille → raw + ranked. |

### `src/HMM.py`
No-lookahead Gaussian HMM -regiimiluokittelija (Bull / Bear / Transition). Oma erillinen ajo. Output: `data/01_raw/outputs/hmm_regimes_monthly_no_lookahead_ml.csv`.

| Funktio | Tehtävä |
|---|---|
| `fetch_input_data(start, end)` | STOXX-hinnat + ECB-tuottokäyrät (AAA, all). |
| `build_feature_matrix(df_equity, df_yields, df_credit)` | Rakentaa HMM-input-featuret (return, volatility, term spread, credit spread). |
| `get_month_end_observation_dates(hmm_df)` | Listaa kuukausilopun observointipäivät. |
| `zscore_series(values)` | Z-skaalaus. |
| `make_regime_map(raw_states, train_df)` | Mappaa raaka HMM-statet (0,1,2) ekonomiseen järjestykseen (0=Bull, 1=Bear, 2=Transition) ekonomisen scoren perusteella → IDt stabiileja refittien yli. |
| `map_raw_posterior_to_regime_probabilities(raw_posterior, regime_map)` | Soveltaa map:n posteriori-todennäköisyyksiin. |
| `fit_predict_month_end_regime(train_df, features)` | Fittää HMM:n trainiin, ennustaa kuukausilopun regiimin. |
| `select_train_window(hmm_df, source_date)` | Train-ikkuna alusta `source_date`:hen (expanding window, ei lookaheadia). |
| `build_no_lookahead_monthly_regimes(hmm_df, features)` | Pää: per kuukausi refit + ennuste → koko aikasarja regiimejä. |
| `build_ml_safe_monthly_regimes(monthly)` | Lopullinen ML-turvallinen versio (regiimit lagattu, jotta saatavissa kuukausilopussa t). |
| `plot_month_end_feature_regimes(...)` | Diagnostiikkakuvat. |
| `plot_monthly_regime_probabilities(...)` | Posteriori-todennäköisyyksien plot. |
| `main()` | CLI-entrypoint. |

### `src/composites.py`
Composite-signaalien rakennus (Z-keskiarvo tai PCA per ryhmä). **Ei käytössä** nykyisessä putkessa — featuret syötetään malleille yksittäin.

| Funktio | Tehtävä |
|---|---|
| `cross_sectional_zscore(df, features)` | Z-skaalaus per kuukausi. |
| `winsorise(df, features, limits)` | Per-kuukausi 1%/99% klippaus. |
| `aggregate_zscore_avg(df, features, signal_name)` | Keskiarvosta yksi signaali. |
| `aggregate_pca(df, features, signal_name, train_end)` | PCA train-ikkunan datalla → ensimmäinen komponentti. |
| `build_composite_signals(df, method_per_group, train_end)` | Orkestrointi: tekee kaikki composite-signaalit määriteltyjen metodien mukaan. |

### `src/models.py`
Walk-forward training + OOS-evaluointi. Lineaariset (OLS, LASSO, Ridge, ElasticNet) ja puumallit (RF, LightGBM, XGBoost). Two-phase: Phase 1 hyperparametrien valinta, Phase 2 expanding-window kuukausittainen retraining.

| Funktio | Tehtävä |
|---|---|
| **Datanvalmistelu** | |
| `add_forward_return(df, target_col)` | Shiftaa `Excess_Return`:n −1 per stock → next-month target. |
| `_resolve_features(df, feature_cols)` | Suodattaa käytettävissä olevat feature-sarakkeet. |
| `prepare_Xy(df, feature_cols, target_col)` | DataFrame → `X`, `y` numpy-arrayksi. |
| `prepare_X(df, feature_cols)` | Pelkkä `X` (ennustusta varten). |
| **Apurit** | |
| `_mse(y_true, y_pred)` | MSE. |
| `_fit_scaler(X_train)` | StandardScaler trainiin. |
| **Mallikohtaiset tune + fit** | |
| `tune_ols, fit_ols` | OLS (ei tuneja). |
| `tune_lasso, fit_lasso` | LASSO grid-search alfan yli. |
| `tune_ridge, fit_ridge` | Ridge grid-search. |
| `tune_elastic_net, fit_elastic_net` | ElasticNet alfa + l1_ratio grid. |
| `tune_rf, fit_rf` | Random Forest. |
| `tune_lgbm, fit_lgbm` | LightGBM. |
| `tune_xgb, fit_xgb` | XGBoost. |
| `tune_nn, fit_nn` | Feedforward NN — slot varattu, ei toteutusta. |
| **Walk-forward** | |
| `run_walk_forward(...)` | Phase 1 (fixed train+val → hyperparametrit) + Phase 2 (expanding window OOS). |
| **Evaluointi** | |
| `r2_oos(y_true, y_pred, y_benchmark)` | R²_OOS vs. historiallinen keskiarvo. |
| `evaluate_all(predictions_df)` | Per-malli RMSE / MAE / R²_OOS. |
| `main(models, verbose)` | CLI-entrypoint: lataa MASTER, ajaa walk-forwardin, kirjoittaa `predictions_oos.csv` + `evaluation_metrics.csv`. |

### `src/portfolio.py`
Decile long-short -portfoliokonstruktio mallien ennusteista.

| Funktio | Tehtävä |
|---|---|
| `build_portfolio(predictions_df, n_deciles)` | Rankaa stockit per kuukausi predicted excess returnin perusteella → long top decile, short bottom decile, equal-weighted. |
| `compute_sharpe(returns, periods_per_year=12)` | Annualisoitu Sharpe. |
| `compute_cumulative_return(returns)` | Kumulatiivinen tuotto. |
| `compute_max_drawdown(returns)` | Maksimi drawdown. |
| `summarise_performance(portfolio_returns_df)` | Per-malli yhteenvetorivi (Sharpe, cumret, drawdown). |
| `main()` | CLI-entrypoint: lukee `predictions_oos.csv`:n, rakentaa portfolion, tallentaa tuotot ja yhteenvedon. |

---

## Konventiot
- **`src/`** = source of truth, **`notebooks/`** oli historiaa (poistettu siivouksessa).
- **Ei look-aheadia**: kaikki featuret laskettu vain hetken t mennessä saatavilla olevasta datasta. `merge_asof(direction="backward")` valuaatio-fundamenttien spreadissa päivätasolle.
- **Orientaatio**: kaikki valuaatiofeatures masterissa siten että **korkeampi = halvempi = osta-signaali**.
- **`data/`** ja **`results/`** ovat gitignored — ei näy fresh kloonissa.

Katso lisätietoja: `CLAUDE.md` (metodologia), `README.md` (yleiskuva), `Research_Plan_v2.pdf` (akateeminen tausta).
