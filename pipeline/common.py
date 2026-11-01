"""Paths, Wikidata constants and evaluation metrics shared by the pipeline."""
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("AGERATING_DATA", ROOT / "data"))
WD = DATA / "wd"          # Wikidata items, statements, labels, external IDs
TEXT = DATA / "text"      # Wikipedia lead/plot texts
MODELS = ROOT / "models"  # trained weights, one directory per model name
RESULTS = ROOT / "results"
# Wikimedia asks API clients to identify themselves; put a contact (URL or e-mail) in AGERATING_USER_AGENT.
UA = os.environ.get("AGERATING_USER_AGENT", "agerating-pipeline/1.0 (python-urllib)")

BUCKETS = [0, 6, 12, 16, 18]  # suggested minimum ages, German FSK scale
TARGET_PID = "P1981"          # FSK film rating: the training label
TARGET_MAP = {"Q20644794": 0, "Q20644795": 6, "Q20644796": 12, "Q20644797": 16, "Q20644798": 18}
# Decision rule: the estimate is the smallest age whose cumulative probability reaches TAU (rounds up when unsure).
TAU = 0.6
MODEL_NAME = "agerating-lfm3-1.7b-v1"

# Film/TV age-rating properties (video-game, app and music ratings excluded); other countries' ratings are model input.
RATING_PROPS = {
    "P1657": "MPA", "P1981": "FSK", "P2629": "BBFC", "P2684": "Kijkwijzer", "P2747": "Filmiroda", "P2756": "EIRIN",
    "P2758": "CNC_FR", "P2363": "NMHH", "P3156": "ACB", "P3216": "ClassInd", "P3306": "ICAA", "P3402": "CNC_RO",
    "P3428": "INCAA", "P3650": "JMK", "P3818": "KMRB", "P3834": "RTC", "P4437": "FPB", "P5150": "IGAC",
    "P5152": "KAVI", "P5201": "IMDA", "P5970": "Medieradet", "P6452": "CBFC", "P6657": "CHVRS", "P6658": "RCQ",
    "P7327": "OFLC_NZ", "P7573": "IFCO", "P8652": "Netflix", "P8874": "HK", "P8889": "MTRCB", "P9086": "BAMID",
    "P10994": "KFCB", "P11611": "Medietilsynet", "P13773": "LSF", "P14869": "LPF",
}
# The universe: items whose P31 (instance of) is one of these film / TV-show classes. Covers 99% of rated items;
# seasons, episodes and games are excluded.
UNIVERSE_CLASSES = {
    "Q11424": "film", "Q506240": "television film", "Q202866": "animated film", "Q24862": "short film",
    "Q20650540": "anime film", "Q17517379": "animated short film", "Q98807719": "animated television film",
    "Q220898": "original video animation", "Q5398426": "television series", "Q1259759": "miniseries",
    "Q117467246": "animated television series", "Q63952888": "anime television series", "Q526877": "web series",
    "Q15416": "television program", "Q1261214": "television special", "Q581714": "animated series",
}
# Classes that make an item "tv" in the title table (P31 labels mentioning television/series count too, see data.py).
TV_CLASSES = {"Q5398426", "Q1259759", "Q63952888", "Q581714", "Q15416", "Q526877", "Q117467246", "Q21191270",
              "Q1261214", "Q7725310", "Q113687694", "Q17198419"}
# Title fallback for items without an English label: first available label in this language order, else any language.
LABEL_PREF = ["mul", "en", "de", "fr", "es", "it", "pt", "nl", "sv", "pl", "ja", "ko", "zh", "ru"]


def now():
    """UTC timestamp as stored in the database, e.g. 2026-10-31T11:04:18+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metrics(y_true, y_pred):
    """Accuracy figures for bucket indices (0..4): exact, within one step, too low by one/two or more steps."""
    from sklearn.metrics import f1_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    d = y_pred - y_true
    return {
        "acc": float((d == 0).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(BUCKETS)), zero_division=0)),
        "within1": float((np.abs(d) <= 1).mean()),
        "mae_steps": float(np.abs(d).mean()),
        "under": float((d < 0).mean()),
        "under2": float((d <= -2).mean()),
    }
