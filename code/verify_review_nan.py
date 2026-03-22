# -*- coding: utf-8 -*-
"""
验证 review 列中的 NaN/float 问题。
用于确认 TypeError: sequence item 25: expected str instance, float found 的原因。
运行: cd code && python verify_review_nan.py
"""
import pandas as pd
from paths_config import DATA_DIR

DATASETS = ["AM_CDs", "AM_Electronics", "AM_Movies", "TripAdvisor", "Yelp"]

def main():
    for dataset in DATASETS:
        path = f"{DATA_DIR}/{dataset}/train.csv"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[SKIP] {dataset}: 无法读取 {path} - {e}")
            continue

        if "review" not in df.columns:
            print(f"[SKIP] {dataset}: 无 review 列")
            continue

        # 1. review 列中非 str 的统计
        non_str = df["review"].apply(lambda x: not isinstance(x, str))
        n_non_str = non_str.sum()
        if n_non_str > 0:
            print(f"\n=== {dataset} ===")
            print(f"  review 列非 str 的数量: {n_non_str} / {len(df)}")
            non_str_types = df.loc[non_str, "review"].apply(lambda x: type(x).__name__).value_counts()
            for t, c in non_str_types.items():
                print(f"    类型 {t}: {c} 条")
            # 示例
            sample = df.loc[non_str, ["user_idx", "review"]].head(3)
            print("  示例:")
            for _, row in sample.iterrows():
                print(f"    user_idx={row['user_idx']}, review={repr(row['review'])}")

        # 2. 模拟 groupby + join，找出会触发 "sequence item 25" 的用户
        def bad_join(reviews):
            for i, r in enumerate(reviews):
                if not isinstance(r, str):
                    return i, r  # 返回第一个坏元素的下标和值
            return None

        grouped = df.groupby("user_idx")["review"]
        problems = []
        for uid, grp in grouped:
            idx_val = bad_join(grp.tolist())
            if idx_val is not None:
                problems.append((uid, idx_val[0], idx_val[1]))

        if problems:
            print(f"\n  user 嵌入会报错的 user 数量: {len(problems)}")
            print(f"  前 5 个示例 (user_idx, 坏元素下标, 坏元素值):")
            for uid, idx, val in problems[:5]:
                print(f"    user_idx={uid}, 下标={idx} (即第 {idx+1} 个), 值={repr(val)}")
            # 特别关注下标 25
            idx25 = [p for p in problems if p[1] == 25]
            if idx25:
                print(f"  其中下标=25 (第 26 个) 的 user: {[p[0] for p in idx25]}")

    print("\n验证完成。若存在非 str，说明 review 列含 NaN/float，需过滤后再 join。")

if __name__ == "__main__":
    main()
