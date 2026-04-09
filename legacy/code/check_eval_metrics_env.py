#!/usr/bin/env python3
"""检查 METEOR / evaluate 离线缓存是否就绪，避免 FINAL RESULTS 中 METEOR=0。"""
# LEGACY / NOT PART OF THE NEW MAINLINE
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_NLTK = os.path.join(_ROOT, "pretrained_models", "nltk_data")


def main() -> int:
    ok = True
    if os.path.isdir(_NLTK):
        os.environ.setdefault("NLTK_DATA", _NLTK)
        print(f"OK NLTK_DATA -> {_NLTK}")
    else:
        print(f"MISSING 目录不存在: {_NLTK}", file=sys.stderr)
        ok = False

    try:
        import nltk

        for name, rel in (
            ("wordnet", "corpora/wordnet"),
            ("punkt", "tokenizers/punkt"),
            ("omw-1.4", "corpora/omw-1.4"),
        ):
            try:
                nltk.data.find(rel)
                print(f"OK nltk.data {name}")
            except LookupError:
                print(f"MISSING nltk 资源: {name} ({rel})", file=sys.stderr)
                ok = False
    except Exception as e:
        print(f"ERROR import nltk: {e}", file=sys.stderr)
        return 2

    os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
    _cache = os.path.join(_ROOT, "pretrained_models", "evaluate_meteor")
    try:
        import evaluate

        evaluate.load("meteor", cache_dir=_cache)
        print(f"OK evaluate.load('meteor', cache_dir={_cache})")
    except Exception as e:
        print(
            "WARN METEOR 预缓存失败（首次需有网: python -c \"import evaluate; evaluate.load('meteor', cache_dir='...')\"）",
            file=sys.stderr,
        )
        print(f"  详情: {e}", file=sys.stderr)
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
