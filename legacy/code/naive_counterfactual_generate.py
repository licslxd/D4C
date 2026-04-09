# LEGACY / NOT PART OF THE NEW MAINLINE
import os
import sys

_LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_LEGACY_DIR, "..", "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from naive_counterfactual_train import *  # noqa: E402
from paths_config import get_t5_small_dir  # noqa: E402

_t5b = get_t5_small_dir()
_t5_path = _t5b if os.path.exists(_t5b) else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(_t5_path, legacy=True)

tasks = [
    ("AM_Electronics", "AM_CDs"),
    ("AM_Movies", "AM_CDs"),
    ("AM_CDs", "AM_Electronics"),
    ("AM_Movies", "AM_Electronics"),
    ("AM_CDs", "AM_Movies"),
    ("AM_Electronics", "AM_Movies"),
    ("Yelp", "TripAdvisor"),
    ("TripAdvisor", "Yelp")
]