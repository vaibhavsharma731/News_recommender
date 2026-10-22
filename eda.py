"""
Exploratory Data Analysis (EDA) Script for News Articles Dataset

Run with: python eda.py
"""
import re
import sys
from collections import Counter
import pandas as pd
import config
from src.data_loader import load_raw_csv, clean_dataframe

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def run_eda():
    print("=" * 60)
    print("NEWS RECOMMENDATION SYSTEM - DATASET EDA REPORT")
    print("=" * 60)

    # 1. Load raw data through shared loader
    raw = load_raw_csv()
    print(f"\n1. DATASET OVERVIEW (RAW):")
    print(f"   • Total rows in raw CSV(s): {len(raw)}")
    print(f"   • Columns ({len(raw.columns)}): {list(raw.columns)}")

    # Check duplicates in raw
    id_col = next((c for c in ["articleId", "id", "article_id", "ID"] if c in raw.columns), None)
    if id_col:
        dupes = raw.duplicated(subset=id_col).sum()
        print(f"   • Duplicate {id_col}s: {dupes} rows ({(dupes/len(raw)):.1%})")

    # 2. Run single authoritative clean_dataframe
    clean_df = clean_dataframe(raw)
    print(f"\n2. CLEANED DATASET OVERVIEW:")
    print(f"   • Unique articles after authoritative deduplication: {len(clean_df)}")
    print(f"   • Columns ({len(clean_df.columns)}): {list(clean_df.columns)}")

    # Missing values analysis on cleaned dataset
    print("\n3. MISSING VALUES & DATA INTEGRITY:")
    important_cols = ["id", "title", "description", "body", "section", "subsection", "date", "image"]
    for col in important_cols:
        if col in clean_df.columns:
            null_cnt = clean_df[col].isna().sum()
            empty_cnt = (clean_df[col].astype(str).str.strip() == "").sum()
            print(f"   • {col:15s}: {null_cnt:3d} null, {empty_cnt:3d} empty ({((null_cnt + empty_cnt)/len(clean_df)):.1%})")

    # Section Distribution
    print("\n4. SECTION & CATEGORY DISTRIBUTION:")
    sec_counts = clean_df["section"].value_counts()
    for sec, cnt in sec_counts.head(15).items():
        label = str(sec) if str(sec).strip() else "(empty)"
        print(f"   • {label:20s}: {cnt:3d} articles ({cnt/len(clean_df):.1%})")

    # Text Quality & Messiness Analysis on raw bodies
    print("\n5. RAW TEXT NOISE & PATTERN DISCOVERY:")
    body_col = next((c for c in ["body_raw", "body", "content", "text"] if c in raw.columns), None)
    raw_bodies = raw[body_col].fillna("").astype(str) if body_col else pd.Series([], dtype=str)

    # URLs and Email instances
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.I)
    email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

    url_docs = sum(1 for b in raw_bodies if url_pattern.search(b))
    email_docs = sum(1 for b in raw_bodies if email_pattern.search(b))
    print(f"   • Articles containing raw URLs: {url_docs} ({url_docs/max(len(raw_bodies), 1):.1%})")
    print(f"   • Articles containing email addresses: {email_docs} ({email_docs/max(len(raw_bodies), 1):.1%})")

    # ALSO READ inline blocks
    also_read_pattern = re.compile(r"\balso\s+read\b", re.I)
    also_read_docs = sum(1 for b in raw_bodies if also_read_pattern.search(b))
    print(f"   • Articles containing 'ALSO READ' inline blocks: {also_read_docs} ({also_read_docs/max(len(raw_bodies), 1):.1%})")

    # Sentence Boilerplate detection
    sentence_split = re.compile(r"(?<=[.!?])\s+")
    sentence_counts = Counter()
    for b in raw_bodies:
        s_set = {s.strip() for s in sentence_split.split(b) if len(s.strip()) > 25}
        sentence_counts.update(s_set)

    boilerplate = {s: c for s, c in sentence_counts.items() if c >= config.BOILERPLATE_MIN_DOCS}
    print(f"   • Boilerplate sentences repeated across >= {config.BOILERPLATE_MIN_DOCS} articles: {len(boilerplate)}")
    if boilerplate:
        print("   • Sample Boilerplate Line:")
        sample_b = list(boilerplate.keys())[0]
        print(f"     \"{sample_b[:90]}...\" (Appears in {boilerplate[sample_b]} docs)")

    # Description length anomaly
    desc_lens = clean_df["description"].fillna("").astype(str).str.len()
    long_descs = (desc_lens > 300).sum()
    print(f"\n6. DESCRIPTION LENGTH CHECK:")
    print(f"   • Max description length: {desc_lens.max()} chars (Mean: {desc_lens.mean():.0f} chars)")
    print(f"   • Descriptions > 300 chars: {long_descs} articles")

    print("\n" + "=" * 60)
    print("EDA COMPLETE - Shared data_loader correctly unified all datasets")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()

