"""Task 3 features: return-based and sentiment-based, ranked across members each day.

Rule for every feature: the value at row t uses only information available
at the close of trading day t. Each feature is computed per stock, then
ranked among that day's index members and scaled to [-0.5, 0.5]. A missing
rank becomes 0 (neutral). Smoothed versions are an EWMA over time of the
ranked, filled feature.

Features are stored as float32 to halve memory; a rank among about 500
stocks needs only three significant digits.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RETURN_HALFLIVES = [5, 10]        # smoothing half-lives (days) for return features
SENT_HALFLIVES = [5, 21]          # smoothing half-lives (days) for sentiment features
SECTOR_FEATURES = ["rev_5", "rev_21", "mom_12_1"]
# Return features that get smoothed versions. Momentum and volatility are left out:
# they already move slowly, so their smoothed versions were near-copies of the
# originals (correlation 0.97 to 0.99) and were dropped.
RETURN_SMOOTHED = ["rev_1", "rev_5", "rev_21", "rev_5_sec", "rev_21_sec"]
SENT_SMOOTHED = ["sent_bull", "sent_bear", "sent_net", "sent_act"]
SENT_MAP_END = "2010-12-31"       # the sentiment column mapping uses no data after this day


def rank_cs(x, mem):
    """Rank x across members on each day and scale to [-0.5, 0.5]. Missing stays NaN.

    Why: ranks put every feature on the same scale every day and remove
    the influence of extreme values. Uses row t of x and mem only.
    """
    x = x.where(mem)                    # keep members of the SAME day only
    n = x.notna().sum(axis=1)           # how many members have a value that day
    r = x.rank(axis=1)                  # 1 = smallest ... n = largest, ties get the average rank
    den = (n - 1).where(n > 1)          # one value alone cannot be ranked -> NaN
    return r.sub(1).div(den, axis=0) - 0.5


def sector_relative(x, mem, sector):
    """Subtract from x the same-day mean of x among members of the same sector.

    Why: removes moves shared by a whole sector, leaving the stock-specific
    part. sector is a Series stock id -> 2-digit GICS code. Uses row t only.
    """
    x = x.where(mem)                    # keep members of the SAME day only
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for s in sorted(sector.unique()):
        cols = sector.index[sector == s]
        sector_mean = x[cols].mean(axis=1)                      # same-day mean, no time shift
        out.loc[:, cols] = x[cols].sub(sector_mean, axis=0).values
    return out


def ewma_smooth(x, halflife):
    """EWMA over time, per stock: s[t] = a * x[t] + (1 - a) * s[t-1], a = 1 - 0.5**(1/halflife).

    Why: a smoother signal changes more slowly, so positions are held longer.
    Row t depends on rows up to and including t only.
    """
    return x.ewm(halflife=halflife, adjust=False).mean()     # recursive, looks backwards only


def rank_fill_smooth(raw, mem, halflives, smoothed):
    """Rank each raw feature, fill missing ranks with 0, and add the smoothed versions.

    smoothed lists the base features that get one smoothed version per half-life.

    Returns (feats, n_valid): feats maps feature name -> float32 table, and
    n_valid maps base feature name -> number of members with a real
    (not filled) rank on each day, used to report coverage.
    """
    feats, n_valid = {}, {}
    for name in list(raw):
        ranked = rank_cs(raw.pop(name), mem)      # pop frees the raw table once it is ranked
        n_valid[name] = ranked.notna().sum(axis=1)
        filled = ranked.fillna(0.0)               # missing -> neutral
        feats[name] = filled.astype("float32")
        if name in smoothed:
            for h in halflives:
                feats[f"{name}_hl{h}"] = ewma_smooth(filled, h).astype("float32")
    return feats, n_valid


def build_return_features(vadj_c, sig, mem, meta):
    """Build the 18 return-based features from clipped vol-adjusted returns and volatility.

    vadj_c[t] and sig[t] are both known at the close of day t.
    Returns (feats, n_valid), see rank_fill_smooth().
    """
    sector = meta["gics"].astype(str).str[:2]     # first 2 GICS digits = sector
    raw = {}
    raw["rev_1"] = vadj_c                                                   # day t only
    raw["rev_5"] = vadj_c.rolling(5, min_periods=3).mean()                  # days t-4..t
    raw["rev_21"] = vadj_c.rolling(21, min_periods=15).mean()               # days t-20..t
    # rolling(231) at row u covers days u-230..u; shift(21) puts row u = t-21 at row t,
    # so row t covers days t-251..t-21 (the most recent month is skipped)
    raw["mom_12_1"] = vadj_c.rolling(231, min_periods=231).sum().shift(21)
    raw["vol_level"] = np.log(sig)                                          # sig[t]: returns up to day t
    for name in SECTOR_FEATURES:
        raw[name + "_sec"] = sector_relative(raw[name], mem, sector)
    return rank_fill_smooth(raw, mem, RETURN_HALFLIVES, RETURN_SMOOTHED)


def load_sentiment(data_dir="data"):
    """Read bullish and bearish sentiment (rows = CALENDAR days, columns = permno strings)."""
    DATA = Path(data_dir)
    bull = pd.read_csv(DATA / "psych_bullish.csv", index_col="date", parse_dates=True)
    bear = pd.read_csv(DATA / "psych_bearish.csv", index_col="date", parse_dates=True)
    return bull, bear


def sentiment_column_map(sent_columns, ret, sent_start, sent_end=SENT_MAP_END):
    """Decide which equity id each sentiment column (a permno) belongs to.

    Why: 19 permnos exist twice in the equity data (ids 'X' and 'X_b'), and
    the sentiment file does not say which of the two it describes. Rule from
    the group: if exactly one of the two has returns between sent_start and
    sent_end, use that one; if both do, use neither. sent_end is the last
    development day, so the decision never depends on test-period data.

    Returns (col_map, report): col_map maps permno -> equity id for the
    columns that are kept; report maps every duplicated permno to what was done.
    """
    col_map, report = {}, {}
    for p in sent_columns:
        a, b = p, p + "_b"
        if b not in ret.columns:                  # not a duplicated permno
            if a in ret.columns:                  # (absent = security dropped as empty)
                col_map[p] = a
            continue
        has_a = a in ret.columns and ret[a].loc[sent_start:sent_end].notna().any()   # days sent_start..sent_end
        has_b = ret[b].loc[sent_start:sent_end].notna().any()                        # days sent_start..sent_end
        if has_a and has_b:
            report[p] = "excluded: both securities have returns"
        elif has_a:
            col_map[p], report[p] = a, f"mapped to {a}"
        elif has_b:
            col_map[p], report[p] = b, f"mapped to {b}"
        else:
            report[p] = "unused: neither security has returns"
    return col_map, report


def align_sentiment(sent, trading_days):
    """Move calendar-day sentiment onto the trading day where it first becomes usable.

    Timing rule: sentiment dated d may include posts after that day's close,
    so it is usable only at the close of the first trading day STRICTLY AFTER d.
    Friday, Saturday and Sunday all land on Monday and are averaged (NaN = no
    messages is ignored; if all are NaN the result is NaN).
    """
    # position of the first trading day strictly after each calendar day d
    pos = trading_days.searchsorted(sent.index, side="right")
    keep = pos < len(trading_days)                   # no later trading day -> never usable
    usable_day = trading_days[pos[keep]]
    return sent[keep].groupby(usable_day).mean()     # row t holds sentiment dated before t only


def build_sentiment_features(bull, bear, col_map, mem):
    """Build the 13 sentiment features on the trading days of the sentiment period.

    bullish, bearish, net = bullish - bearish, activity = bullish + bearish,
    each ranked across members, plus smoothed versions and a 0/1 flag that
    says whether the stock had any usable sentiment that day.
    Returns (feats, n_valid), see rank_fill_smooth().
    """
    trading_days = mem.index
    days = trading_days[trading_days > bull.index[0]]     # first usable day is after the first sentiment day
    keep = list(col_map)

    def to_equity_table(sent):
        """Align to trading days and relabel permno columns as equity ids."""
        aligned = align_sentiment(sent, trading_days)
        aligned = aligned[keep].rename(columns=col_map)
        return aligned.reindex(index=days, columns=mem.columns)

    b, r = to_equity_table(bull), to_equity_table(bear)
    m = mem.loc[days]
    flag = (b.notna() | r.notna()) & m                     # member with usable sentiment on day t
    raw = {"sent_bull": b, "sent_bear": r,
           "sent_net": b - r,                              # same-day difference, no time shift
           "sent_act": b + r}                              # same-day sum, no time shift
    feats, n_valid = rank_fill_smooth(raw, m, SENT_HALFLIVES, SENT_SMOOTHED)
    feats["sent_flag"] = flag.astype("float32")            # stays 0/1, not ranked
    n_valid["sent_flag"] = flag.sum(axis=1)
    return feats, n_valid
