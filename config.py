"""
Projektin konfiguraatio: universumi, aikavälit, parametrit.
"""

from datetime import datetime, timedelta

# =====================================================================
# Aikaväli
# =====================================================================
# DISPLAY_START_DATE = lopullisen paneelin alku (mitä mallille syötetään)
# FETCH_START_DATE   = API-haun alku (warmup mukana, jotta Beta_252d /
#                      Hurst ehtivät kypsyä ennen DISPLAY_START_DATE:ä)
#
# Warmup mitoitetaan kalenteripäivinä: 252 trading day ≈ ~365 kalenteripäivää,
# + buffer pyhäpäiviä ja datan katkoja varten → 450 kalenteripäivää riittää.
DISPLAY_START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

WARMUP_CALENDAR_DAYS = 600
FETCH_START_DATE = (
    datetime.strptime(DISPLAY_START_DATE, "%Y-%m-%d")
    - timedelta(days=WARMUP_CALENDAR_DAYS)
).strftime("%Y-%m-%d")

# Taaksepäin yhteensopiva alias (moduulit jotka odottavat START_DATE-nimeä)
START_DATE = DISPLAY_START_DATE

# Refinitiv-parametrit (päivätaajuus, EUR) — HAKU sisältää warmupin
PARAMS_DAILY = {
    "SDate": FETCH_START_DATE,
    "EDate": END_DATE,
    "Frq": "D",
    "Curn": "EUR",
}

# EURIBOR ei käytä Curn-parametria
PARAMS_EURIBOR = {
    "SDate": FETCH_START_DATE,
    "EDate": END_DATE,
    "Frq": "D",
}

# Indeksit
INDEX_UNIVERSE = [".STOXX", ".STOXXR"]

# EURIBOR RIC
EURIBOR_RIC = ["EURIBOR3MD="]

# API chunk-koko (Refinitiv-rajoitus)
CHUNK_SIZE = 50
SLEEP_BETWEEN_CHUNKS = 1  # sekuntia

# Rolling-ikkunat
BETA_WINDOW = 252
VOL_WINDOW = 30
RSI_WINDOW = 30
HURST_WINDOW = 252
MOM_1M_WINDOW = 21   # ~1 kuukausi kaupankäyntipäiviä
MOM_12M_WINDOW = 252  # ~12 kuukautta kaupankäyntipäiviä
MOM_12M_SKIP = 21     # skip viimeisin kuukausi (t-1)

# DEPRECATED — vanhat 4 binäärisen FF-tyylisen GICS-sektoridummyn arvot.
# Korvattu Sector_Group-aggregaatiolla (GICS 11 → 6 eurooppalaiselle universumille).
# Säilytetty tyhjinä listoina, jotta vanhat callerit (main.py, main_test.py,
# updated_main_test.py) saavat importin läpi. compute_sector_dummies on no-op.
# Vanhat arvot:
# SECTOR_DUMMIES = ["Consumer Discretionary", "Industrials", "Information Technology", "Health Care"]
# SECTOR_DUMMY_NAMES = ["Cnsmr", "Manuf", "HiTec", "Hlth"]
SECTOR_DUMMIES = []
SECTOR_DUMMY_NAMES = []

# Walk-forward ikkunat
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2018-12-31"
OOS_START = "2019-01-01"
OOS_END = "2025-12-31"
