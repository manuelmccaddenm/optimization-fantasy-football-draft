"""Construye el universo de jugadores, mu y Sigma desde datos crudos y escribe los CSV."""
import os
import re as _re
import numpy as np
import pandas as pd
from typing import Dict
from optimizacion import LeagueConfig, HyperParams, POSITIONS

POS_CV = {"QB": 0.40, "RB": 0.55, "WR": 0.55, "TE": 0.55, "K": 0.45, "DST": 0.50}
CRUDOS = "datos_crudos"
NFLVERSE_CACHE = os.path.join(CRUDOS, "nflverse_cache")
NFLVERSE_YEARS = [2021, 2022, 2023, 2024]
ADP_CSV = os.path.join(CRUDOS, "adp_2025.csv")
ROOKIES_CSV = os.path.join(CRUDOS, "rookies_2025.csv")
INJURY_CSV = os.path.join(CRUDOS, "injury_risk_2026.csv")
KICKERS_CSV = os.path.join(CRUDOS, "kickers_2025.csv")
ROOKIES_2026 = set()


def _norm_name(s: str) -> str:
    s = _re.sub(r"[^a-z ]", "", str(s).lower())
    for suf in (" jr", " sr", " ii", " iii", " iv", " v"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


def _load_rookies_2026(path: str = "rookies_2026.csv") -> set:
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path)["name"].tolist())
    except Exception:
        return set()


def load_real_players(csv_path: str, league: LeagueConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if os.path.exists(KICKERS_CSV):
        kdf = pd.read_csv(KICKERS_CSV)
        kdf = kdf.sort_values("adp").iloc[20:].reset_index(drop=True)
        df = pd.concat([df[df["pos"] != "K"], kdf], ignore_index=True)

    fa_mask = df["team"] == "FA"
    df.loc[fa_mask, "team"] = [f"FA{i:02d}" for i in range(fa_mask.sum())]

    pos_mu_range = {
        "QB": (200, 380), "RB": (50, 360), "WR": (50, 380),
        "TE": (40, 280), "K": (90, 180), "DST": (60, 180),
    }
    df["mu"] = 0.0
    for pos in pos_mu_range:
        sub = df[df["pos"] == pos].sort_values("adp")
        n = len(sub)
        if n == 0:
            continue
        lo, hi = pos_mu_range[pos]
        ranks = np.arange(n)
        mus = hi * np.exp(-ranks / max(n, 1) * 1.6)
        df.loc[sub.index, "mu"] = np.clip(mus, lo * 0.5, hi)

    extras = []
    last_adp = df["adp"].max()
    replacement_mu = {"QB": 180, "RB": 70, "WR": 70, "TE": 60, "K": 95, "DST": 90}
    for p in POSITIONS:
        cur = (df["pos"] == p).sum()
        need = max(0, league.n_teams * league.pos_max[p] - cur)
        for i in range(need):
            extras.append({
                "name": f"Generic_{p}{i+1}",
                "pos": p,
                "team": f"GEN{p}{i:02d}",
                "adp": last_adp + 10 + i + 50 * POSITIONS.index(p),
                "mu": replacement_mu.get(p, 80.0),
            })
    if extras:
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)

    df = df.reset_index(drop=True)
    df["player_id"] = np.arange(len(df))
    return df


def build_real_mu_sigma(df: pd.DataFrame, hp: HyperParams,
                         cache_dir: str = NFLVERSE_CACHE,
                         years=NFLVERSE_YEARS, min_overlap: int = 10):
    frames = []
    for y in years:
        p = os.path.join(cache_dir, f"ps_{y}.csv")
        d = pd.read_csv(p, low_memory=False)
        d = d[(d["season_type"] == "REG") &
              (d["position"].isin(["QB", "RB", "WR", "TE"]))]
        frames.append(d[["player_display_name", "season", "week",
                          "fantasy_points_ppr"]])
        kp = os.path.join(cache_dir, f"kick_{y}.csv")
        if os.path.exists(kp):
            kd = pd.read_csv(kp, low_memory=False)
            kd = kd[kd["season_type"] == "REG"].rename(
                columns={"fantasy_points": "fantasy_points_ppr"})
            frames.append(kd[["player_display_name", "season", "week",
                              "fantasy_points_ppr"]])
    W = pd.concat(frames, ignore_index=True)
    W["wk"] = W["season"].astype(str) + "_" + W["week"].astype(str).str.zfill(2)
    M = W.pivot_table(index="player_display_name", columns="wk",
                      values="fantasy_points_ppr", aggfunc="mean")
    M.index = [_norm_name(x) for x in M.index]
    M = M.groupby(level=0).mean()

    col_season = np.array([int(c.split("_")[0]) for c in M.columns])
    seasons_uniq = np.unique(col_season)
    # mu usa solo semanas jugadas (talento); Sigma usa la rejilla de 17 con
    # ceros en los partidos no disputados, para que la lesion entre al riesgo.
    M_played = M.copy()

    Mg = M.values.copy()
    for ridx in range(Mg.shape[0]):
        row = Mg[ridx]
        for s in seasons_uniq:
            cmask = col_season == s
            seg = row[cmask]
            if np.any(~np.isnan(seg)):
                seg[np.isnan(seg)] = 0.0
                row[cmask] = seg
    M_grid = pd.DataFrame(Mg, index=M.index, columns=M.columns)

    latest = col_season.max()
    col_w = hp.mu_recency_decay ** (latest - col_season)
    Pv = M_played.values
    obs = ~np.isnan(Pv)
    wsum = (obs * col_w[None, :]).sum(axis=1)
    weighted = np.nansum(np.where(obs, Pv, 0.0) * col_w[None, :], axis=1)
    mu_arr = np.where(wsum > 0, weighted / np.maximum(wsum, 1e-9) * 17, np.nan)
    mu_real = dict(zip(M_played.index, mu_arr))

    cov = (M_grid.T.cov() * 17.0)
    mask = M_grid.notna().astype(float)
    overlap = mask @ mask.T
    cov = cov.where(overlap >= min_overlap, 0.0)
    real_idx = list(M_grid.index)
    real_pos = {i: cov.index.get_loc(i) for i in real_idx}
    covv = cov.values

    n = len(df)
    mu_v = df["mu"].values.astype(float).copy()
    pos = df["pos"].values
    Sig = np.zeros((n, n))
    norm_names = [_norm_name(x) for x in df["name"].values]
    row_of = {}
    matched = 0
    w = hp.mu_hist_weight

    real_sig_by_pos = {p: [] for p in POSITIONS}
    for k, nm in enumerate(norm_names):
        if nm in real_pos:
            hist = mu_real.get(nm, np.nan)
            if np.isfinite(hist):
                mu_v[k] = w * hist + (1.0 - w) * mu_v[k]
            rk = real_pos[nm]
            row_of[k] = rk
            matched += 1
            v = covv[rk, rk]
            if v > 0:
                real_sig_by_pos[pos[k]].append(np.sqrt(v))

    pos_emp_sigma = {}
    for p in POSITIONS:
        arr = real_sig_by_pos[p]
        pos_emp_sigma[p] = float(np.median(arr)) if arr else None

    def _prior_sigma(k):
        base = pos_emp_sigma.get(pos[k])
        if base is None or base <= 0:
            base = POS_CV.get(pos[k], 0.5) * mu_v[k]
        return max(base, 1.0)

    for k in range(n):
        if k in row_of:
            rk = row_of[k]
            for l, rl in row_of.items():
                Sig[k, l] = covv[rk, rl]
            if Sig[k, k] <= 0:
                Sig[k, k] = _prior_sigma(k) ** 2
        else:
            Sig[k, k] = _prior_sigma(k) ** 2

    is_rk = df["name"].isin(ROOKIES_2026).values
    mu_v = np.where(is_rk, mu_v * hp.rookie_mu_factor, mu_v)

    # Contraccion suave a la diagonal y proyeccion al cono PSD (la covarianza
    # por pares no es semidefinida positiva por construccion).
    Sig = 0.5 * (Sig + Sig.T)
    d = np.diag(np.diag(Sig))
    Sig = 0.90 * Sig + 0.10 * d
    vals, vecs = np.linalg.eigh(Sig)
    vals = np.clip(vals, 1e-6 * max(vals.max(), 1.0), None)
    Sig = (vecs * vals) @ vecs.T
    return mu_v, Sig, matched


def apply_rookie_variance(Sigma: np.ndarray, df: pd.DataFrame,
                           rookies: set, mult: float) -> np.ndarray:
    Sigma = Sigma.copy()
    is_rookie = df["name"].isin(rookies).values
    scale = np.where(is_rookie, np.sqrt(mult), 1.0)
    return Sigma * scale[:, None] * scale[None, :]


def _load_injury_risk(path: str = "injury_risk_2026.csv") -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        d = pd.read_csv(path)
        return dict(zip(d["name"], d["injury_risk"].astype(float)))
    except Exception:
        return {}


def apply_injury_variance(Sigma: np.ndarray, df: pd.DataFrame,
                           injury: Dict[str, float], max_var_mult: float) -> np.ndarray:
    Sigma = Sigma.copy()
    risk_arr = df["name"].map(lambda n: injury.get(n, 0.0)).fillna(0.0).values.astype(float)
    sigma_scale = 1.0 + risk_arr * (np.sqrt(max_var_mult) - 1.0)
    return Sigma * sigma_scale[:, None] * sigma_scale[None, :]


def main():
    global ROOKIES_2026
    league, hp = LeagueConfig(), HyperParams()
    ROOKIES_2026 = _load_rookies_2026(ROOKIES_CSV)
    players = load_real_players(ADP_CSV, league)
    mu, Sigma, n = build_real_mu_sigma(players, hp)
    injury = _load_injury_risk(INJURY_CSV)
    if os.path.exists(ROOKIES_CSV):
        rk = pd.read_csv(ROOKIES_CSV)
        if "injury_risk" in rk.columns:
            injury.update(dict(zip(rk["name"],
                                   rk["injury_risk"].astype(float))))
    Sigma = apply_rookie_variance(Sigma, players, ROOKIES_2026,
                                  hp.rookie_var_mult)
    Sigma = apply_injury_variance(Sigma, players, injury,
                                  hp.injury_max_var_mult)
    players["mu_hat"] = mu
    players["sigma_hat"] = np.sqrt(np.diag(Sigma))
    players["is_rookie"] = players["name"].isin(ROOKIES_2026)
    os.makedirs("data", exist_ok=True)
    players.to_csv("data/jugadores.csv", index=False)
    np.savetxt("data/mu.csv", mu, delimiter=",")
    np.savetxt("data/sigma.csv", Sigma, delimiter=",")
    print(f"jugadores={len(players)}  emparejados={n}  "
          f"Sigma={Sigma.shape}  -> data/")


if __name__ == "__main__":
    main()
