"""
Exploratory Data Analysis (EDA) Script for News Articles Dataset

Run with: python eda.py
"""
import json
import re
import sys
from collections import Counter
import pandas as pd
import config

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def run_eda():
    print("=" * 60)
    print("NEWS RECOMMENDATION SYSTEM - DATASET EDA REPORT")

    print("=" * 60)
    
    csv_path = config.DATA_PATH
    raw = pd.read_csv(csv_path)
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   • Total rows in raw CSV: {len(raw)}")
    print(f"   • Columns ({len(raw.columns)}): {list(raw.columns)}")
    
    # Check duplicates
    dupes = raw.duplicated(subset="articleId").sum()
    print(f"   • Duplicate articleIds: {dupes} rows ({(dupes/len(raw)):.1%})")
    
    clean_df = raw.drop_duplicates(subset="articleId", keep="first").copy()
    print(f"   • Unique articles after deduplication: {len(clean_df)}")
    
    # Missing values analysis
    print("\n2. MISSING VALUES & DATA INTEGRITY:")
    important_cols = ["articleId", "articleTitle", "description", "body_raw", "sectionName", "section2Name", "postPublishDate", "articleFeaturedImage"]
    for col in important_cols:
        if col in clean_df.columns:
            null_cnt = clean_df[col].isna().sum()
            print(f"   • {col:25s}: {null_cnt:3d} missing ({null_cnt/len(clean_df):.1%})")
            
    # Section Distribution
    print("\n3. SECTION & CATEGORY DISTRIBUTION:")
    sec_counts = clean_df["sectionName"].value_counts()
    for sec, cnt in sec_counts.items():
        print(f"   • {str(sec):20s}: {cnt:3d} articles ({cnt/len(clean_df):.1%})")
        
    # Text Quality & Messiness Analysis
    print("\n4. TEXT NOISE & PATTERN DISCOVERY:")
    bodies = clean_df["body_raw"].fillna("").astype(str)
    
    # URLs and Email instances
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.I)
    email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
    
    url_docs = sum(1 for b in bodies if url_pattern.search(b))
    email_docs = sum(1 for b in bodies if email_pattern.search(b))
    print(f"   • Articles containing raw URLs: {url_docs} ({url_docs/len(clean_df):.1%})")
    print(f"   • Articles containing email addresses: {email_docs} ({email_docs/len(clean_df):.1%})")
    
    # ALSO READ inline blocks
    also_read_pattern = re.compile(r"\balso\s+read\b", re.I)
    also_read_docs = sum(1 for b in bodies if also_read_pattern.search(b))
    print(f"   • Articles containing 'ALSO READ' inline blocks: {also_read_docs} ({also_read_docs/len(clean_df):.1%})")
    
    # Sentence Boilerplate detection
    sentence_split = re.compile(r"(?<=[.!?])\s+")
    sentence_counts = Counter()
    for b in bodies:
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
    print(f"\n5. DESCRIPTION LENGTH ANOMALIES:")
    print(f"   • Max description length: {desc_lens.max()} chars (Mean: {desc_lens.mean():.0f} chars)")
    print(f"   • Descriptions > 300 chars: {long_descs} articles (Needs clipping to avoid noise)")

    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE - All insights incorporated into src/data_loader.py")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
