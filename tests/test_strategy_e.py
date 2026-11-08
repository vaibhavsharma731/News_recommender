"""
Tests for Strategy E — Asymmetric Section and Subsection Boosting
==================================================================
Covers:
1. Same umbrella section and same subsection.
2. Same umbrella section but different subsection.
3. Same focused section and same subsection.
4. Different sections.
5. Empty subsection.
6. Case and whitespace normalization.
7. Unknown section handling (safe 1.020 / 1.020 fallback).
"""

import sys, os
import math
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.scoring import boost_parts

D1 = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
D_MAX = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)

def test_same_umbrella_same_subsection():
    """1. Same umbrella section and same subsection -> sec=1.020, sub=1.020."""
    src = {"id": "src1", "section": "cities", "subsection": "Mumbai", "date": D1}
    cand = {"id": "c1", "section": "cities", "subsection": "Mumbai", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.020, rel_tol=1e-5), f"Expected 1.020, got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.020, rel_tol=1e-5), f"Expected 1.020, got {parts['subsection']}"
    print("PASS: test_same_umbrella_same_subsection")

def test_same_umbrella_diff_subsection():
    """2. Same umbrella section but different subsection -> sec=1.020, sub=1.000."""
    src = {"id": "src2", "section": "india", "subsection": "Politics", "date": D1}
    cand = {"id": "c2", "section": "india", "subsection": "Crime", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.020, rel_tol=1e-5), f"Expected 1.020, got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.000, rel_tol=1e-5), f"Expected 1.000, got {parts['subsection']}"
    print("PASS: test_same_umbrella_diff_subsection")

def test_same_focused_same_subsection():
    """3. Same focused section and same subsection -> sec=1.100, sub=1.050."""
    src = {"id": "src3", "section": "sports", "subsection": "Cricket", "date": D1}
    cand = {"id": "c3", "section": "sports", "subsection": "Cricket", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.100, rel_tol=1e-5), f"Expected 1.100, got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.050, rel_tol=1e-5), f"Expected 1.050, got {parts['subsection']}"
    print("PASS: test_same_focused_same_subsection")

def test_different_sections():
    """4. Different sections -> sec=1.000, sub=1.000 (even if subsections match name)."""
    src = {"id": "src4", "section": "sports", "subsection": "Cricket", "date": D1}
    cand = {"id": "c4", "section": "entertainment", "subsection": "Cricket", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.000, rel_tol=1e-5), f"Expected 1.000, got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.000, rel_tol=1e-5), f"Expected 1.000, got {parts['subsection']}"
    print("PASS: test_different_sections")

def test_empty_subsection():
    """5. Empty subsection -> sec boost applies, sub=1.000."""
    src = {"id": "src5", "section": "sports", "subsection": "", "date": D1}
    cand = {"id": "c5", "section": "sports", "subsection": "", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.100, rel_tol=1e-5), f"Expected 1.100, got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.000, rel_tol=1e-5), f"Expected 1.000, got {parts['subsection']}"

    # Also test None
    src_none = {"id": "src5b", "section": "cities", "subsection": None, "date": D1}
    cand_none = {"id": "c5b", "section": "cities", "subsection": None, "date": D1}
    parts_none = boost_parts(src_none, cand_none, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts_none["section"], 1.020, rel_tol=1e-5), f"Expected 1.020, got {parts_none['section']}"
    assert math.isclose(parts_none["subsection"], 1.000, rel_tol=1e-5), f"Expected 1.000, got {parts_none['subsection']}"
    print("PASS: test_empty_subsection")

def test_case_and_whitespace_normalization():
    """6. Case and whitespace normalization."""
    src = {"id": "src6", "section": "  SpOrTs  ", "subsection": "  CrIcKeT  ", "date": D1}
    cand = {"id": "c6", "section": "sports", "subsection": "cricket", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.100, rel_tol=1e-5), f"Expected 1.100, got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.050, rel_tol=1e-5), f"Expected 1.050, got {parts['subsection']}"

    # Umbrella normalization test
    src_u = {"id": "src6u", "section": "  InDiA\t", "subsection": "  PoLiTiCs\n", "date": D1}
    cand_u = {"id": "c6u", "section": "india", "subsection": "politics", "date": D1}
    parts_u = boost_parts(src_u, cand_u, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts_u["section"], 1.020, rel_tol=1e-5), f"Expected 1.020, got {parts_u['section']}"
    assert math.isclose(parts_u["subsection"], 1.020, rel_tol=1e-5), f"Expected 1.020, got {parts_u['subsection']}"
    print("PASS: test_case_and_whitespace_normalization")

def test_unknown_section_handling():
    """7. Unknown section handling defaults to conservative Umbrella (1.020 / 1.020)."""
    src = {"id": "src7", "section": "quantum-computing", "subsection": "hardware", "date": D1}
    cand = {"id": "c7", "section": "quantum-computing", "subsection": "hardware", "date": D1}
    
    parts = boost_parts(src, cand, max_views=1000, max_date=D_MAX)
    assert math.isclose(parts["section"], 1.020, rel_tol=1e-5), f"Expected 1.020 (conservative fallback), got {parts['section']}"
    assert math.isclose(parts["subsection"], 1.020, rel_tol=1e-5), f"Expected 1.020 (conservative fallback), got {parts['subsection']}"
    print("PASS: test_unknown_section_handling")

if __name__ == "__main__":
    test_same_umbrella_same_subsection()
    test_same_umbrella_diff_subsection()
    test_same_focused_same_subsection()
    test_different_sections()
    test_empty_subsection()
    test_case_and_whitespace_normalization()
    test_unknown_section_handling()
    print("\nAll 7 Strategy E tests passed successfully!")

