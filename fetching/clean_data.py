"""
Clean the UFC master dataset.

Reads ufc-master.csv and produces ufc-clean.csv with the following changes,
applied in order:

1. Drop unwanted columns (win breakdowns, derived diffs, rankings, fight
   outcomes, raw odds).  See `fetching/columns.md` for the full KEEP/DROP rationale.
2. Drop rows with missing or invalid betting odds.
3. Convert American moneyline odds to decimal odds and normalized implied
   probabilities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT = PROJECT_ROOT / "datasets/ultimate-ufc/ufc-master.csv"
OUTPUT = PROJECT_ROOT / "datasets/ultimate-ufc/ufc-clean.csv"

if not INPUT.exists():
    print(f"Error: {INPUT} not found", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(INPUT)
print(f"Loaded {INPUT} — {df.shape[0]} rows, {df.shape[1]} columns")


# ── 1. Drop unwanted columns ────────────────────────────────────────────────

drop_cols = [
    # Win breakdown: how each fighter won their past fights.
    # Granularity not needed — total Wins/Losses already kept.
    "RedWinsByDecisionMajority", "RedWinsByDecisionSplit",
    "RedWinsByDecisionUnanimous", "RedWinsByKO",
    "RedWinsBySubmission", "RedWinsByTKODoctorStoppage",
    "BlueWinsByDecisionMajority", "BlueWinsByDecisionSplit",
    "BlueWinsByDecisionUnanimous", "BlueWinsByKO",
    "BlueWinsBySubmission", "BlueWinsByTKODoctorStoppage",

    # Pre-computed Blue-minus-Red difference features.
    # Redundant — can be derived from the raw Red/Blue columns.
    "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif", "WinDif",
    "LossDif", "TotalRoundDif", "TotalTitleBoutDif", "KODif", "SubDif",
    "HeightDif", "ReachDif", "AgeDif", "SigStrDif", "AvgSubAttDif", "AvgTDDif",

    # Rankings: per-division rank columns are ~97% null (only top-15 fighters
    # are ranked), and the consolidated RMatchWCRank/BMatchWCRank are still
    # 73-82% null.  Too sparse to be useful.
    "RMatchWCRank", "BMatchWCRank", "RPFPRank", "BPFPRank", "BetterRank",
    "RWFlyweightRank", "RWFeatherweightRank", "RWStrawweightRank",
    "RWBantamweightRank", "RHeavyweightRank", "RLightHeavyweightRank",
    "RMiddleweightRank", "RWelterweightRank", "RLightweightRank",
    "RFeatherweightRank", "RBantamweightRank", "RFlyweightRank",
    "BWFlyweightRank", "BWFeatherweightRank", "BWStrawweightRank",
    "BWBantamweightRank", "BHeavyweightRank", "BLightHeavyweightRank",
    "BMiddleweightRank", "BWelterweightRank", "BLightweightRank",
    "BFeatherweightRank", "BBantamweightRank", "BFlyweightRank",

    # Fight outcome columns: Finish, FinishDetails, FinishRound, etc.
    # These describe HOW the fight ended — using them to predict WHO wins
    # would be target leakage.
    "Finish", "FinishDetails", "FinishRound", "FinishRoundTime",
    "TotalFightTimeSecs",

    # Raw odds columns: replaced in step 3 by decimal odds and normalized
    # implied probabilities.  Method-specific odds (decision/sub/KO) and
    # ExpectedValue (a 1:1 transform of moneyline) are also dropped.
    "RedOdds", "BlueOdds", "RedExpectedValue", "BlueExpectedValue",
    "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds",
]

df = df.drop(columns=drop_cols)
print(f"After dropping columns — {df.shape[1]} columns remain")


# ── 2. Drop rows with missing or invalid odds ───────────────────────────────
#
# Some rows have no odds at all (~227 rows).  Others have odds where both
# fighters are listed as favorites (both negative moneyline), or the implied
# probabilities sum to less than 1 (negative vig).  These indicate bad or
# stale data scraped from different times/sources.
#
# We filter to rows where the raw implied probabilities sum to a reasonable
# range: between 1.0 and 1.15 (a typical bookmaker vig is 2-8%).

raw = pd.read_csv(INPUT, usecols=["RedOdds", "BlueOdds"])


def moneyline_to_raw_prob(odds: pd.Series) -> pd.Series:
    """Convert American moneyline to raw (non-normalized) implied probability."""
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100), 100 / (odds + 100))


red_raw = moneyline_to_raw_prob(raw["RedOdds"])
blue_raw = moneyline_to_raw_prob(raw["BlueOdds"])
prob_sum = red_raw + blue_raw

has_odds = raw["RedOdds"].notna() & raw["BlueOdds"].notna()
valid_vig = (prob_sum >= 1.0) & (prob_sum <= 1.15)
keep_mask = has_odds & valid_vig

n_before = len(df)
mask = keep_mask.values
df = df[mask].reset_index(drop=True)
red_raw = pd.Series(red_raw[mask]).reset_index(drop=True)
blue_raw = pd.Series(blue_raw[mask]).reset_index(drop=True)
prob_sum = pd.Series(prob_sum[mask]).reset_index(drop=True)
raw = raw[mask].reset_index(drop=True)
print(f"After filtering invalid odds — dropped {n_before - len(df)} rows, {len(df)} remain")


# ── 3. Convert odds to decimal format and normalized probabilities ───────────
#
# Decimal odds: the payout multiplier (e.g. 2.5 means $1 returns $2.50).
#   -250 American  ->  1.40 decimal
#   +215 American  ->  3.15 decimal
#
# Implied probability: the true win probability after removing the bookmaker's
# vig.  Each fighter's raw implied prob is divided by the sum of both, so the
# pair sums to exactly 1.0.

def moneyline_to_decimal(odds: pd.Series) -> pd.Series:
    """Convert American moneyline to decimal odds (multiplier)."""
    return np.where(odds < 0, 1 + 100 / np.abs(odds), 1 + odds / 100)


red_dec = moneyline_to_decimal(raw["RedOdds"])
blue_dec = moneyline_to_decimal(raw["BlueOdds"])

red_prob = red_raw / prob_sum
blue_prob = blue_raw / prob_sum

df.insert(2, "RedDecimalOdds", red_dec)
df.insert(3, "BlueDecimalOdds", blue_dec)
df.insert(4, "RedImpliedProb", red_prob)
df.insert(5, "BlueImpliedProb", blue_prob)


# ── Save ─────────────────────────────────────────────────────────────────────

df.to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT} — {df.shape[0]} rows, {df.shape[1]} columns")
