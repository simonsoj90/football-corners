from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import json
import numpy as np


@dataclass
class ZStats:
    mean_p_home: float
    std_p_home: float
    mean_p_over25: float
    std_p_over25: float


def _ensure_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def load_artifacts(prefix: str | Path):
    pref = Path(prefix)
    npz_path = Path(str(pref) + "_draws.npz")
    teams_path = Path(str(pref) + "_teams.json")
    divs_path = Path(str(pref) + "_divs.json")
    zstats_path = Path(str(pref) + "_zstats.json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing draws file: {npz_path}")
    if not teams_path.exists():
        raise FileNotFoundError(f"Missing teams file: {teams_path}")
    if not zstats_path.exists():
        raise FileNotFoundError(f"Missing zstats file: {zstats_path}")

    npz = np.load(npz_path, allow_pickle=True)
    draws = {k: npz[k] for k in npz.files}

    team_names = json.loads(teams_path.read_text(encoding="utf-8"))
    div_names = json.loads(divs_path.read_text(encoding="utf-8")) if divs_path.exists() else []

    zs = json.loads(zstats_path.read_text(encoding="utf-8"))
    std_p_home = max(_ensure_float(zs.get("std_p_home", 0.1)), 1e-3)
    std_p_over25 = max(_ensure_float(zs.get("std_p_over25", 0.1)), 1e-3)

    zstats = ZStats(
        mean_p_home=_ensure_float(zs.get("mean_p_home", 0.5)),
        std_p_home=std_p_home,
        mean_p_over25=_ensure_float(zs.get("mean_p_over25", 0.5)),
        std_p_over25=std_p_over25,
    )
    return draws, team_names, div_names, zstats


def parse_odds_decimal(x) -> Optional[float]:
    
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            return v if v > 1.0 else None
        except Exception:
            return None
    s = str(x).strip().lower()
    if s in ("", "na", "n/a", "nan", "-", "--"):
        return None
    if s in ("even", "evens"):
        return 2.0
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            num, den = float(num.strip()), float(den.strip())
            if den <= 0:
                return None
            return 1.0 + num / den
        except Exception:
            return None
    try:
        v = float(s)
        return v if v > 1.0 else None
    except Exception:
        return None


def implied_probs_1x2_from_odds(oh: float, od: float, oa: float) -> Tuple[float, float, float]:
    
    inv = np.array([1.0/oh, 1.0/od, 1.0/oa], dtype=float)
    s = inv.sum()
    if not np.isfinite(s) or s <= 0:
        return 1/3, 1/3, 1/3
    p = inv / s
    return float(p[0]), float(p[1]), float(p[2])


def implied_prob_over25_from_odds(o_over: float, o_under: float) -> float:
    
    inv = np.array([1.0/o_over, 1.0/o_under], dtype=float)
    s = inv.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.5
    p = inv / s
    return float(p[0])


def zscore(x: float, mean_: float, std_: float) -> float:
    return (float(x) - float(mean_)) / (float(std_) + 1e-8)


def nb_predictive_samples(mu_draws: np.ndarray,
                          alpha_draws: np.ndarray,
                          n_sims: int = 10000) -> np.ndarray:

    mu = np.asarray(mu_draws).reshape(-1)
    alpha = np.asarray(alpha_draws).reshape(-1)

    mu = np.clip(np.nan_to_num(mu, nan=10.0, posinf=60.0, neginf=1e-6), 1e-6, 60.0)
    alpha = np.clip(np.nan_to_num(alpha, nan=20.0, posinf=1e3, neginf=1e-6), 1e-6, 1e3)

    D = len(mu)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, D, size=n_sims)
    lam = rng.gamma(shape=alpha[idx], scale=(mu[idx] / alpha[idx]))
    lam = np.clip(lam, 0.0, 1e6)  # final guard against overflow
    y = rng.poisson(lam)
    return y.astype(int)


def _find_idx(name: str, universe: List[str]) -> int:
    try:
        return universe.index(name)
    except ValueError:
        return 0


def predict_game(draws: Dict[str, np.ndarray],
                 team_names: List[str],
                 div_names: List[str],
                 zstats: ZStats,
                 home_team: str,
                 away_team: str,
                 division: Optional[str],
                 odds_home: float, odds_draw: float, odds_away: float,
                 odds_over25: float, odds_under25: float,
                 total_corners_line: float,
                 price_over: Optional[float] = None,
                 price_under: Optional[float] = None,
                 n_sims: int = 10000) -> Dict[str, float]:

    hi = _find_idx(home_team, team_names)
    ai = _find_idx(away_team, team_names)
    if isinstance(division, str) and len(div_names) > 0:
        try:
            di = div_names.index(division)
        except ValueError:
            di = 0
    else:
        di = 0

    p_home, p_draw, p_away = implied_probs_1x2_from_odds(odds_home, odds_draw, odds_away)
    p_over25 = implied_prob_over25_from_odds(odds_over25, odds_under25)

    z_ph = float(np.clip(zscore(p_home,  zstats.mean_p_home,  zstats.std_p_home),  -6.0, 6.0))
    z_po = float(np.clip(zscore(p_over25, zstats.mean_p_over25, zstats.std_p_over25), -6.0, 6.0))

    intercept = draws["intercept"].reshape(-1)
    home_adv  = draws["home_adv"].reshape(-1)
    team_home_eff = draws["team_home_eff"]
    team_away_eff = draws["team_away_eff"]
    alpha         = draws["alpha"].reshape(-1)

    beta_homeprob = draws.get("beta_homeprob", np.zeros_like(intercept)).reshape(-1)
    beta_over25   = draws.get("beta_over25",   np.zeros_like(intercept)).reshape(-1)

    if "div_eff" in draws and len(div_names) > 0:
        div_term = draws["div_eff"][:, di].reshape(-1)
    else:
        div_term = 0.0

    eta = (intercept + home_adv
           + team_home_eff[:, hi] + team_away_eff[:, ai]
           + div_term
           + beta_homeprob * z_ph + beta_over25 * z_po)

    mu = np.exp(eta)
    mu = np.clip(mu, 1e-6, 60.0)

    sims = nb_predictive_samples(mu, alpha, n_sims=n_sims)

    # quick .5-style (no push) probabilities w.r.t. 'line'
    thresh = int(np.floor(total_corners_line + 1e-9)) + 1
    p_over = float(np.mean(sims >= thresh))
    p_under = 1.0 - p_over

    fair_over  = 1.0 / max(p_over, 1e-9)
    fair_under = 1.0 / max(p_under, 1e-9)

    res = {
        "expected_corners": float(np.mean(sims)),
        "p_over_line": p_over,
        "p_under_line": p_under,
        "fair_odds_over": fair_over,
        "fair_odds_under": fair_under,
    }
    if price_over is not None:
        res["edge_over"] = (price_over / fair_over) - 1.0
    if price_under is not None:
        res["edge_under"] = (price_under / fair_under) - 1.0
    return res


def pmf_from_draws(sims: np.ndarray, vmax: Optional[int] = None) -> Dict[int, float]:
    vals, counts = np.unique(sims, return_counts=True)
    pmf = {int(v): float(c/len(sims)) for v, c in zip(vals, counts)}
    if vmax is not None:
        pmf = {k: v for k, v in pmf.items() if k <= vmax}
    return pmf


def prob_over_under_push(pmf: Dict[int, float], line: float) -> Tuple[float, float, float]:

    if abs(line - round(line)) < 1e-9:
        L = int(round(line))
        p_over = sum(p for k, p in pmf.items() if k > L)
        p_under = sum(p for k, p in pmf.items() if k < L)
        p_push = pmf.get(L, 0.0)
    else:
        thresh = int(np.floor(line + 1e-9)) + 1
        p_over = sum(p for k, p in pmf.items() if k >= thresh)
        p_under = 1.0 - p_over
        p_push = 0.0
    return float(p_over), float(p_under), float(p_push)


def asian_quarter_split(line: float) -> List[Tuple[float, float]]:

    frac = round((line - np.floor(line)) % 1.0, 2)
    if abs(frac - 0.25) < 1e-9:
        return [(np.floor(line), 0.5), (np.floor(line) + 0.5, 0.5)]
    if abs(frac - 0.75) < 1e-9:
        return [(np.floor(line) + 0.5, 0.5), (np.floor(line) + 1.0, 0.5)]
    return [(line, 1.0)]


def ev_kelly_two_way(p_win: float,
                     p_lose: float,
                     price: Optional[float],
                     kelly_fraction: float = 0.5) -> Tuple[float, float]:

    if price is None or not np.isfinite(price) or price <= 1.0:
        return np.nan, 0.0
    b = float(price) - 1.0
    ev = p_win * b - p_lose
    k_full = (b * p_win - p_lose) / b
    k = max(0.0, k_full) * float(kelly_fraction)
    return float(ev), float(k)


def ev_kelly_single(p: float,
                    price: Optional[float],
                    kelly_fraction: float = 0.5) -> Tuple[float, float]:

    if price is None or not np.isfinite(price) or price <= 1.0:
        return np.nan, 0.0
    b = float(price) - 1.0
    q = 1.0 - float(p)
    k_full = (b * p - q) / b
    ev = p * b - q
    k = max(0.0, k_full) * float(kelly_fraction)
    return float(ev), float(k)

def implied_prob_two_way(price_over: float, price_under: float) -> float:

    import numpy as np
    inv = np.array([1.0 / price_over, 1.0 / price_under], dtype=float)
    s = inv.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.5
    return float(inv[0] / s)

def gaussian_copula_joint(pA: float, pB: float, rho: float = 0.2,
                          n: int = 200_000, seed: int = 42) -> float:

    import numpy as np
    rho = float(np.clip(rho, -0.95, 0.95))
    if abs(rho) < 1e-12:
        return float(pA * pB)
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal(size=n)
    from math import erf, sqrt
    Phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    u1 = Phi(z1)
    u2 = Phi(z2)
    return float(np.mean((u1 <= pA) & (u2 <= pB)))

