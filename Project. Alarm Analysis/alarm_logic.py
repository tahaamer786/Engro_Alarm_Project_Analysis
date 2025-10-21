# alarm_logic.py
import pandas as pd
from datetime import timedelta

# Configurable thresholds
STALE_MINUTES = 30
CHATTER_WINDOW_SECONDS = 60       # repeated alarms within 1 minute => chattering
UNHEALTHY_COUNT = 10              # >= 10 in 10 minutes => unhealthy
FLOOD_WINDOW_MINUTES = 10
FLOOD_THRESHOLD = 50              # total alarms across sources in window => Flood

def ensure_datetime(df, time_col="Event Time"):
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    return df

def classify_per_source(df, time_col="Event Time"):
    df = ensure_datetime(df, time_col)
    results = []

    # Group by Source
    for src, g in df.groupby("Source"):
        g = g.sort_values(time_col).reset_index(drop=True)
        total = len(g)

        # time diffs
        g['time_diff'] = g[time_col].diff()

        # Active (no ACK/OK in the record). We treat Action column; empty or NaN -> active count
        active_count = g[g['Action'].isna() | (g['Action'].astype(str).str.strip() == "")].shape[0]

        # Stale: time_diff > STALE_MINUTES
        stale_count = g[g['time_diff'] > timedelta(minutes=STALE_MINUTES)].shape[0]

        # Chattering: successive alarms within CHATTER_WINDOW_SECONDS (and possibly actions present)
        chatter_count = g[g['time_diff'] <= timedelta(seconds=CHATTER_WINDOW_SECONDS)].shape[0]

        # Unhealthy in last window logic handled outside (see rolling window)
        results.append({
            "Source": src,
            "Total Alarms": total,
            "Active Count": int(active_count),
            "Stale Count": int(stale_count),
            "Chattering Count": int(chatter_count)
        })

    return pd.DataFrame(results)

def detect_unhealthy_and_badactor(df, time_col="Event Time"):
    df = ensure_datetime(df, time_col)
    # Unhealthy: sources with >= UNHEALTHY_COUNT in FLOOD_WINDOW_MINUTES rolling window
    window = f"{FLOOD_WINDOW_MINUTES}T"
    temp = df.set_index(time_col)
    counts_by_time = temp.resample(window)['Source'].value_counts().unstack(fill_value=0)
    # flatten to find sources that in any window meet threshold
    unhealthy = {}
    for src in counts_by_time.columns:
        if (counts_by_time[src] >= UNHEALTHY_COUNT).any():
            unhealthy[src] = int(counts_by_time[src].max())

    # Bad actor: source(s) with max total count overall
    total_counts = df['Source'].value_counts()
    if not total_counts.empty:
        max_count = int(total_counts.max())
        bad_actors = total_counts[total_counts == max_count].index.tolist()
    else:
        bad_actors = []
        max_count = 0

    return unhealthy, {"bad_actors": bad_actors, "max_count": max_count}

def detect_floods(df, time_col="Event Time"):
    df = ensure_datetime(df, time_col)
    window = f"{FLOOD_WINDOW_MINUTES}T"
    temp = df.set_index(time_col)
    rolling_counts = temp['Source'].resample(window).count()
    floods = rolling_counts[rolling_counts >= FLOOD_THRESHOLD]
    # return list of (window_start, count)
    return [{"window_start": str(idx), "count": int(c)} for idx, c in floods.items()]

def analyze(df, time_col="Event Time"):
    """
    Main entry. Returns dict summary with per-source classification and flood/bad-actor info.
    """
    df = df.copy()
    df = ensure_datetime(df, time_col)
    per_source = classify_per_source(df, time_col)
    unhealthy, badactor = detect_unhealthy_and_badactor(df, time_col)
    floods = detect_floods(df, time_col)

    return {
        "per_source": per_source.to_dict(orient="records"),
        "unhealthy_sources": unhealthy,
        "bad_actor": badactor,
        "floods": floods,
        "total_rows": int(len(df))
    }
