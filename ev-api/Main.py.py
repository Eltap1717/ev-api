# main.py
import os
import time
import math
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Callable

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# EV-ENGINE (SPORT-AGNOSTIC) — unchanged structure
# ============================================================

def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)

def calculate_ev(probability: float, odds: float) -> Dict[str, float]:
    implied = 1.0 / odds
    edge = probability - implied
    return {
        "probability": probability,
        "implied_probability": implied,
        "edge": edge,
        "ev": probability * odds - 1.0,
        "edge_percent": edge * 100.0,
    }


# ============================================================
# INPUT NORMALIZATION (Loveable-friendly)
# ============================================================

_SELECTION_ALIASES = {
    "o": "over",
    "over": "over",
    "u": "under",
    "under": "under",
    "y": "yes",
    "yes": "yes",
    "n": "no",
    "no": "no",
    "1": "home",
    "home": "home",
    "x": "draw",
    "draw": "draw",
    "2": "away",
    "away": "away",
    "to_score": "yes",
    "goal": "yes",
    "assist": "yes",
}

def normalize_selection(selection: str) -> str:
    s = (selection or "").lower().strip()
    return _SELECTION_ALIASES.get(s, s)

def normalize_team_side(team_side: Optional[str]) -> Optional[str]:
    if team_side is None:
        return None
    s = team_side.lower().strip()
    if s in ("h", "home"):
        return "home"
    if s in ("a", "away"):
        return "away"
    return s


# ============================================================
# NUMERICALLY STABLE DISTRIBUTIONS
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, k + 1):
        s += poisson_pmf(i, lam)
    return clamp01(s)

def negbin_pmf(k: int, lam: float, alpha: float) -> float:
    """
    Negative Binomial via Poisson-Gamma mixture (mean=lam, dispersion=alpha).
    alpha: larger -> more Poisson-like.
    """
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    if alpha <= 0.0:
        return poisson_pmf(k, lam)

    p = alpha / (alpha + lam)
    return math.exp(
        math.lgamma(k + alpha) - math.lgamma(alpha) - math.lgamma(k + 1)
        + alpha * math.log(p)
        + k * math.log(1.0 - p)
    )

def negbin_cdf(k: int, lam: float, alpha: float) -> float:
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, k + 1):
        s += negbin_pmf(i, lam, alpha)
    return clamp01(s)

def prob_over_nb(line: float, lam: float, alpha: float) -> float:
    threshold = math.floor(line) + 1
    return 1.0 - negbin_cdf(threshold - 1, lam, alpha)

def prob_under_nb(line: float, lam: float, alpha: float) -> float:
    threshold = math.floor(line)
    return negbin_cdf(threshold, lam, alpha)

def prob_over_poisson(line: float, lam: float) -> float:
    threshold = math.floor(line) + 1
    return 1.0 - poisson_cdf(threshold - 1, lam)

def prob_under_poisson(line: float, lam: float) -> float:
    threshold = math.floor(line)
    return poisson_cdf(threshold, lam)


# ============================================================
# CACHE (TTL + MAX ITEMS + PER-KEY LOCK)
# ============================================================

class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 2048):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, Tuple[float, float, Any]] = {}
        self._lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            lk = self._key_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._key_locks[key] = lk
            return lk

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, _, value = item
            if now > expires_at:
                self._store.pop(key, None)
                return None
            self._store[key] = (expires_at, now, value)
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            if len(self._store) >= self.max_items:
                oldest_key = None
                oldest_access = float("inf")
                for k, (_, last_access, _) in self._store.items():
                    if last_access < oldest_access:
                        oldest_access = last_access
                        oldest_key = k
                if oldest_key is not None:
                    self._store.pop(oldest_key, None)
            self._store[key] = (now + self.ttl, now, value)

    def get_or_set(self, key: str, fn: Callable[[], Any]):
        cached = self.get(key)
        if cached is not None:
            return cached
        lk = self._get_key_lock(key)
        with lk:
            cached2 = self.get(key)
            if cached2 is not None:
                return cached2
            val = fn()
            self.set(key, val)
            return val


# ============================================================
# BAYESIAN SHRINKAGE
# ============================================================

def shrink_mean(observed_mean: float, n: float, prior_mean: float, prior_strength: float) -> float:
    n = max(0.0, float(n))
    prior_strength = max(0.0, float(prior_strength))
    if n <= 0.0:
        return prior_mean
    w = n / (n + prior_strength) if (n + prior_strength) > 0 else 1.0
    return w * observed_mean + (1.0 - w) * prior_mean


# ============================================================
# API-FOOTBALL CLIENT (CACHED + RETRIES)
# ============================================================

@dataclass
class ApiFootballConfig:
    base_url: str
    api_key_env: str = "API_FOOTBALL_KEY"
    timeout_seconds: int = 10
    cache_ttl_seconds: int = 300
    cache_max_items: int = 4000

class ApiFootballClient:
    def __init__(self, cfg: ApiFootballConfig):
        self.cfg = cfg
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set env var {cfg.api_key_env}.")
        self.api_key = api_key
        self.cache = TTLCache(ttl_seconds=cfg.cache_ttl_seconds, max_items=cfg.cache_max_items)

    def _headers(self) -> Dict[str, str]:
        return {
            "x-apisports-key": self.api_key,
            "Accept": "application/json",
        }

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/" + path.lstrip("/")
        cache_key = f"GET:{url}:{json.dumps(params, sort_keys=True, default=str)}"

        def _fetch():
            retries = 2
            base_backoff = 1.6
            last_err: Optional[str] = None

            for attempt in range(retries + 1):
                try:
                    resp = requests.get(
                        url,
                        headers=self._headers(),
                        params=params,
                        timeout=self.cfg.timeout_seconds,
                    )

                    if resp.status_code == 429 and attempt < retries:
                        ra = resp.headers.get("Retry-After")
                        if ra and ra.isdigit():
                            time.sleep(int(ra))
                        else:
                            time.sleep(base_backoff ** attempt)
                        continue

                    if resp.status_code >= 400:
                        last_err = f"{resp.status_code} {resp.text[:200]}"
                    else:
                        return resp.json()

                except requests.exceptions.RequestException as e:
                    last_err = str(e)
                    if attempt < retries:
                        time.sleep(base_backoff ** attempt)

            raise HTTPException(status_code=502, detail=f"API-Football request failed: {last_err or 'Unknown error'}")

        return self.cache.get_or_set(cache_key, _fetch)


# ============================================================
# LEAGUES & SEASON
# ============================================================

DEFAULT_FOOTBALL_LEAGUES = {
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "eredivisie": 88,
    "primera_division": 2,
}

def load_leagues() -> Dict[str, int]:
    raw = os.getenv("FOOTBALL_LEAGUES_JSON")
    if not raw:
        return DEFAULT_FOOTBALL_LEAGUES
    try:
        obj = json.loads(raw)
        return {str(k): int(v) for k, v in obj.items()}
    except Exception:
        return DEFAULT_FOOTBALL_LEAGUES

def infer_season() -> int:
    env = os.getenv("FOOTBALL_SEASON")
    if env and env.isdigit():
        return int(env)
    now = time.gmtime()
    return now.tm_year if now.tm_mon >= 7 else now.tm_year - 1


# ============================================================
# FOOTBALL MODEL
# ============================================================

class FootballModel:
    def __init__(self, client: ApiFootballClient, leagues: Dict[str, int]):
        self.client = client
        self.leagues = leagues
        self.referee_cache = TTLCache(ttl_seconds=86400, max_items=512)

        self.prior = {
            "team_goals_mean": 1.35,
            "team_goals_strength": 10.0,
            "team_xgproxy_mean": 1.45,
            "team_xgproxy_strength": 10.0,
            "total_goals_mean": 2.70,
            "total_goals_strength": 12.0,

            "corners_total_mean": 10.0,
            "corners_total_strength": 12.0,
            "cards_total_mean": 4.4,
            "cards_total_strength": 12.0,
            "offsides_total_mean": 4.0,
            "offsides_total_strength": 12.0,

            "shots_total_mean": 11.0,      # per TEAM
            "shots_total_strength": 10.0,
            "sot_total_mean": 4.0,         # per TEAM
            "sot_total_strength": 10.0,

            "player_shots90_mean": 1.60,
            "player_shots90_strength": 6.0,
            "player_sot90_mean": 0.70,
            "player_sot90_strength": 6.0,
            "player_goal_mean": 0.22,
            "player_goal_strength": 6.0,
            "player_assist_mean": 0.15,
            "player_assist_strength": 6.0,
        }

    # ---------- Helpers ----------

    def _safe_float(self, x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str) and x.strip() != "":
                return float(x)
            return None
        except Exception:
            return None

    def _response_list(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        resp = data.get("response")
        return resp if isinstance(resp, list) else []

    def _league_id(self, league_slug: str) -> int:
        league_slug = league_slug.lower().strip()
        if league_slug not in self.leagues:
            raise HTTPException(status_code=400, detail=f"Unsupported league slug: {league_slug}")
        return self.leagues[league_slug]

    def _fixture(self, fixture_id: int) -> Dict[str, Any]:
        data = self.client.get("fixtures", {"id": fixture_id})
        resp = self._response_list(data)
        if not resp:
            raise HTTPException(status_code=400, detail="fixture_id not found")
        return resp[0]

    def _fixture_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        data = self.client.get("fixtures/statistics", {"fixture": fixture_id})
        return self._response_list(data)

    def _team_recent_fixtures(self, team_id: int, league_id: int, season: int, last_n: int) -> List[Dict[str, Any]]:
        data = self.client.get("fixtures", {"team": team_id, "league": league_id, "season": season, "last": last_n})
        return self._response_list(data)

    def _player_season_stats(self, league_id: int, season: int, player_id: int) -> Dict[str, Any]:
        data = self.client.get("players", {"id": player_id, "league": league_id, "season": season})
        resp = self._response_list(data)
        return resp[0] if resp else {}

    def _teams_from_fixture(self, fixture: Dict[str, Any]) -> Tuple[int, int]:
        teams = fixture.get("teams") or {}
        home = (teams.get("home") or {}).get("id")
        away = (teams.get("away") or {}).get("id")
        if not home or not away:
            raise HTTPException(status_code=400, detail="Could not extract team ids from fixture")
        return int(home), int(away)

    def _fixture_score(self, fixture: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        goals = fixture.get("goals") or {}
        try:
            h = goals.get("home")
            a = goals.get("away")
            return (int(h) if h is not None else None, int(a) if a is not None else None)
        except Exception:
            return None, None

    def _stat_from_stats_block(self, stats: List[Dict[str, Any]], stat_type: str, side: str) -> Optional[float]:
        if not stats:
            return None
        entry = stats[0] if side == "home" else (stats[1] if len(stats) > 1 else stats[0])
        s_list = entry.get("statistics")
        if not isinstance(s_list, list):
            return None
        for s in s_list:
            if not isinstance(s, dict):
                continue
            if str(s.get("type", "")).lower() == stat_type.lower():
                return self._safe_float(s.get("value"))
        return None

    # ---------- Bayesian team profiles ----------

    def _team_attack_defense_profile_bayes(
        self,
        team_id: int,
        league_id: int,
        season: int,
        last_n: int,
        inputs: Dict[str, Any],
    ) -> Tuple[float, float, float, float, List[str]]:
        notes: List[str] = []
        fixtures = self._team_recent_fixtures(team_id, league_id, season, last_n)
        if not fixtures:
            return (
                self.prior["team_goals_mean"],
                self.prior["team_goals_mean"],
                self.prior["team_xgproxy_mean"],
                self.prior["team_xgproxy_mean"],
                ["No recent fixtures; using priors."],
            )

        gf = ga = 0.0
        xg_for = xg_against = 0.0
        cnt = 0

        for f in fixtures:
            fid = (f.get("fixture") or {}).get("id")
            if not fid:
                continue

            stats = None
            try:
                stats = self._fixture_stats(int(fid))
            except Exception:
                stats = None

            hg, ag = self._fixture_score(f)
            teams = f.get("teams") or {}
            home_id = (teams.get("home") or {}).get("id")
            away_id = (teams.get("away") or {}).get("id")
            if not home_id or not away_id:
                continue

            is_home = int(home_id) == team_id
            is_away = int(away_id) == team_id
            if not (is_home or is_away):
                continue

            if hg is not None and ag is not None:
                if is_home:
                    gf += hg
                    ga += ag
                else:
                    gf += ag
                    ga += hg

            if stats:
                if is_home:
                    sh_for = self._stat_from_stats_block(stats, "Total Shots", "home") or 0.0
                    sot_for = self._stat_from_stats_block(stats, "Shots on Goal", "home") or 0.0
                    sh_against = self._stat_from_stats_block(stats, "Total Shots", "away") or 0.0
                    sot_against = self._stat_from_stats_block(stats, "Shots on Goal", "away") or 0.0
                else:
                    sh_for = self._stat_from_stats_block(stats, "Total Shots", "away") or 0.0
                    sot_for = self._stat_from_stats_block(stats, "Shots on Goal", "away") or 0.0
                    sh_against = self._stat_from_stats_block(stats, "Total Shots", "home") or 0.0
                    sot_against = self._stat_from_stats_block(stats, "Shots on Goal", "home") or 0.0

                c_sot = float(inputs.get("xgproxy_coef_sot", 0.08))
                c_sh = float(inputs.get("xgproxy_coef_sh", 0.03))
                xg_for += c_sot * sot_for + c_sh * sh_for
                xg_against += c_sot * sot_against + c_sh * sh_against

            cnt += 1

        if cnt <= 0:
            return (
                self.prior["team_goals_mean"],
                self.prior["team_goals_mean"],
                self.prior["team_xgproxy_mean"],
                self.prior["team_xgproxy_mean"],
                ["No usable fixtures; using priors."],
            )

        gf_obs = gf / cnt
        ga_obs = ga / cnt
        xgf_obs = xg_for / cnt
        xga_obs = xg_against / cnt

        gf_mu = shrink_mean(gf_obs, cnt, self.prior["team_goals_mean"], float(inputs.get("team_goals_strength", self.prior["team_goals_strength"])))
        ga_mu = shrink_mean(ga_obs, cnt, self.prior["team_goals_mean"], float(inputs.get("team_goals_strength", self.prior["team_goals_strength"])))
        xgf_mu = shrink_mean(xgf_obs, cnt, self.prior["team_xgproxy_mean"], float(inputs.get("team_xgproxy_strength", self.prior["team_xgproxy_strength"])))
        xga_mu = shrink_mean(xga_obs, cnt, self.prior["team_xgproxy_mean"], float(inputs.get("team_xgproxy_strength", self.prior["team_xgproxy_strength"])))

        notes.append(f"Team profile Bayes: n={cnt}, gf {gf_obs:.2f}->{gf_mu:.2f}, ga {ga_obs:.2f}->{ga_mu:.2f}, xg {xgf_obs:.2f}->{xgf_mu:.2f}.")
        return gf_mu, ga_mu, xgf_mu, xga_mu, notes

    # ---------- Expected goals fixture ----------

    def _expected_goals_fixture(
        self,
        fixture_id: int,
        league_slug: str,
        inputs: Dict[str, Any],
    ) -> Tuple[float, float, List[str]]:
        notes: List[str] = []
        league_id = self._league_id(league_slug)
        season = infer_season()

        fixture = self._fixture(fixture_id)
        home_id, away_id = self._teams_from_fixture(fixture)

        # caps for Render/API stability
        last_n = int(inputs.get("last_n", 10))
        last_n = max(3, min(last_n, 10))

        hg_f, hg_a, hxg_f, hxg_a, n1 = self._team_attack_defense_profile_bayes(home_id, league_id, season, last_n, inputs)
        ag_f, ag_a, axg_f, axg_a, n2 = self._team_attack_defense_profile_bayes(away_id, league_id, season, last_n, inputs)
        notes += n1 + n2

        base_lambda_home = float(inputs.get("base_lambda_home", 1.40))
        base_lambda_away = float(inputs.get("base_lambda_away", 1.10))
        home_adv = float(inputs.get("home_adv", 1.10))
        away_penalty = float(inputs.get("away_penalty", 0.95))

        goals_prior = float(inputs.get("team_goals_prior_mean", self.prior["team_goals_mean"]))
        xg_prior = float(inputs.get("team_xg_prior_mean", self.prior["team_xgproxy_mean"]))

        atk_home = 0.5 * (hg_f / max(goals_prior, 0.2)) + 0.5 * (hxg_f / max(xg_prior, 0.2))
        def_home = 0.5 * (hg_a / max(goals_prior, 0.2)) + 0.5 * (hxg_a / max(xg_prior, 0.2))
        atk_away = 0.5 * (ag_f / max(goals_prior, 0.2)) + 0.5 * (axg_f / max(xg_prior, 0.2))
        def_away = 0.5 * (ag_a / max(goals_prior, 0.2)) + 0.5 * (axg_a / max(xg_prior, 0.2))

        lam_home = base_lambda_home * atk_home * (1.0 / max(def_away, 0.35)) * home_adv
        lam_away = base_lambda_away * atk_away * (1.0 / max(def_home, 0.35)) * away_penalty

        lam_home = max(0.20, float(lam_home))
        lam_away = max(0.20, float(lam_away))

        total_obs = lam_home + lam_away
        total_mu = shrink_mean(
            total_obs,
            float(inputs.get("total_goals_nproxy", last_n)),
            float(inputs.get("total_goals_prior", self.prior["total_goals_mean"])),
            float(inputs.get("total_goals_strength", self.prior["total_goals_strength"])),
        )
        scale = total_mu / max(total_obs, 1e-6)
        lam_home *= scale
        lam_away *= scale

        notes.append(f"λ_home={lam_home:.2f}, λ_away={lam_away:.2f} (Bayes hybrid, total shrunk).")
        return lam_home, lam_away, notes

    # ---------- Match markets ----------

    def prob_1x2(self, fixture_id: int, league_slug: str, selection: str, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        # cap max_goals for runtime stability
        max_goals = int(inputs.get("max_goals", 8))
        max_goals = max(5, min(max_goals, 8))

        lam_home, lam_away, notes = self._expected_goals_fixture(fixture_id, league_slug, inputs)

        p_home = p_draw = p_away = 0.0
        for hg in range(0, max_goals + 1):
            ph = poisson_pmf(hg, lam_home)
            for ag in range(0, max_goals + 1):
                pa = poisson_pmf(ag, lam_away)
                p = ph * pa
                if hg > ag:
                    p_home += p
                elif hg == ag:
                    p_draw += p
                else:
                    p_away += p

        sel = normalize_selection(selection)
        if sel in ("home",):
            return clamp01(p_home), "football_1x2_bayes_hybrid", notes
        if sel in ("draw",):
            return clamp01(p_draw), "football_1x2_bayes_hybrid", notes
        if sel in ("away",):
            return clamp01(p_away), "football_1x2_bayes_hybrid", notes
        raise HTTPException(status_code=400, detail="Invalid selection for match_winner_1x2 (home/draw/away)")

    def prob_goals_ou(self, fixture_id: int, league_slug: str, selection: str, line: float, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        lam_home, lam_away, notes = self._expected_goals_fixture(fixture_id, league_slug, inputs)
        lam_total = lam_home + lam_away
        alpha = float(inputs.get("dispersion", 12.0))

        sel = normalize_selection(selection)
        if sel == "over":
            p = prob_over_nb(line, lam_total, alpha)
        elif sel == "under":
            p = prob_under_nb(line, lam_total, alpha)
        else:
            raise HTTPException(status_code=400, detail="Invalid selection for goals_ou (over/under)")

        notes.append(f"Totals NB: λ={lam_total:.2f}, α={alpha:.1f}.")
        return clamp01(p), "football_goals_ou_nb_bayes", notes

    def prob_btts(self, fixture_id: int, league_slug: str, selection: str, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        lam_home, lam_away, notes = self._expected_goals_fixture(fixture_id, league_slug, inputs)

        p_home_score = 1.0 - math.exp(-lam_home)
        p_away_score = 1.0 - math.exp(-lam_away)
        p_yes = p_home_score * p_away_score

        sel = normalize_selection(selection)
        if sel == "yes":
            notes.append("BTTS approx assumes independent scoring processes (Poisson-style).")
            return clamp01(p_yes), "football_btts_yes_bayes", notes
        if sel == "no":
            notes.append("BTTS approx assumes independent scoring processes (Poisson-style).")
            return clamp01(1.0 - p_yes), "football_btts_no_bayes", notes
        raise HTTPException(status_code=400, detail="Invalid selection for btts (yes/no)")

    def prob_team_goals_ou(self, fixture_id: int, league_slug: str, selection: str, line: float, team_side: str, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        lam_home, lam_away, notes = self._expected_goals_fixture(fixture_id, league_slug, inputs)
        side = normalize_team_side(team_side) or ""
        if side not in ("home", "away"):
            raise HTTPException(status_code=400, detail="team_side must be 'home' or 'away'")

        lam = lam_home if side == "home" else lam_away
        alpha = float(inputs.get("dispersion", 10.0))

        sel = normalize_selection(selection)
        if sel == "over":
            p = prob_over_nb(line, lam, alpha)
        elif sel == "under":
            p = prob_under_nb(line, lam, alpha)
        else:
            raise HTTPException(status_code=400, detail="Invalid selection for team_goals_ou (over/under)")

        notes.append(f"Team goals NB: side={side}, λ={lam:.2f}, α={alpha:.1f}.")
        return clamp01(p), f"football_team_goals_ou_{side}_nb_bayes", notes

    # ---------- Per-team fixture stat helpers (fixed for team_shots/team_sot correctness) ----------

    def _avg_fixture_stat_for_team(self, team_id: int, league_id: int, season: int, last_n: int, stat_type_name: str) -> Tuple[float, int]:
        fixtures = self._team_recent_fixtures(team_id, league_id, season, last_n)
        total = 0.0
        cnt = 0
        for f in fixtures:
            fid = (f.get("fixture") or {}).get("id")
            if not fid:
                continue
            try:
                stats = self._fixture_stats(int(fid))
            except Exception:
                continue

            teams = f.get("teams") or {}
            home = (teams.get("home") or {}).get("id")
            away = (teams.get("away") or {}).get("id")
            if not home or not away:
                continue

            side = "home" if int(home) == team_id else ("away" if int(away) == team_id else None)
            if side is None:
                continue

            val = self._stat_from_stats_block(stats, stat_type_name, side)
            if val is None:
                continue
            total += float(val)
            cnt += 1

        return (total / cnt, cnt) if cnt > 0 else (0.0, 0)

    def prob_team_stat_ou(
        self,
        fixture_id: int,
        league_slug: str,
        selection: str,
        line: float,
        team_side: str,
        stat_type_name: str,
        prior_mean_team: float,
        prior_strength_team: float,
        inputs: Dict[str, Any],
        model_suffix: str,
    ) -> Tuple[float, str, List[str]]:
        notes: List[str] = []
        league_id = self._league_id(league_slug)
        season = infer_season()
        fixture = self._fixture(fixture_id)
        home_id, away_id = self._teams_from_fixture(fixture)

        side = normalize_team_side(team_side) or ""
        if side not in ("home", "away"):
            raise HTTPException(status_code=400, detail="team_side must be 'home' or 'away'")

        team_id = home_id if side == "home" else away_id

        last_n = int(inputs.get("last_n", 8))
        last_n = max(3, min(last_n, 10))

        obs_mean, obs_n = self._avg_fixture_stat_for_team(team_id, league_id, season, last_n, stat_type_name)
        n_proxy = float(obs_n)

        lam = shrink_mean(
            obs_mean if obs_mean > 0 else prior_mean_team,
            n_proxy,
            float(inputs.get("prior_mean", prior_mean_team)),
            float(inputs.get("prior_strength", prior_strength_team)),
        )

        notes.append(f"{stat_type_name}({side}): obs λ={obs_mean:.2f} (n={obs_n}) -> shrunk λ={lam:.2f}.")

        alpha = float(inputs.get("dispersion", 10.0))
        sel = normalize_selection(selection)
        if sel == "over":
            p = prob_over_nb(line, lam, alpha)
        elif sel == "under":
            p = prob_under_nb(line, lam, alpha)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid selection for {model_suffix} (over/under)")

        notes.append(f"NB: α={alpha:.1f}.")
        return clamp01(p), f"football_{model_suffix}_ou_nb_bayes", notes

    # ---------- Match-total stats (corners/cards/offsides remain match totals as before) ----------

    def prob_fixture_stats_ou(
        self,
        fixture_id: int,
        league_slug: str,
        selection: str,
        line: float,
        stat_type_name: str,
        prior_mean_total: float,
        prior_strength_total: float,
        inputs: Dict[str, Any],
        model_suffix: str,
    ) -> Tuple[float, str, List[str]]:
        notes: List[str] = []
        league_id = self._league_id(league_slug)
        season = infer_season()
        fixture = self._fixture(fixture_id)
        home_id, away_id = self._teams_from_fixture(fixture)

        last_n = int(inputs.get("last_n", 8))
        last_n = max(3, min(last_n, 10))

        home_mean, home_n = self._avg_fixture_stat_for_team(home_id, league_id, season, last_n, stat_type_name)
        away_mean, away_n = self._avg_fixture_stat_for_team(away_id, league_id, season, last_n, stat_type_name)

        lam_obs = (home_mean + away_mean)
        n_proxy = float(min(home_n, away_n)) if (home_n > 0 and away_n > 0) else float(max(home_n, away_n))

        lam = shrink_mean(
            lam_obs if lam_obs > 0 else prior_mean_total,
            n_proxy,
            float(inputs.get("prior_mean", prior_mean_total)),
            float(inputs.get("prior_strength", prior_strength_total)),
        )

        notes.append(f"{stat_type_name}(match): obs λ={lam_obs:.2f} (n~{n_proxy:.1f}) -> shrunk λ={lam:.2f}.")

        alpha = float(inputs.get("dispersion", 10.0))
        sel = normalize_selection(selection)
        if sel == "over":
            p = prob_over_nb(line, lam, alpha)
        elif sel == "under":
            p = prob_under_nb(line, lam, alpha)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid selection for {stat_type_name} OU (over/under)")

        notes.append(f"NB: α={alpha:.1f}.")
        return clamp01(p), f"football_{model_suffix}_ou_nb_bayes", notes

    def prob_corners_ou(self, fixture_id: int, league_slug: str, selection: str, line: float, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        return self.prob_fixture_stats_ou(
            fixture_id, league_slug, selection, line,
            stat_type_name="Corner Kicks",
            prior_mean_total=float(inputs.get("default_lambda", self.prior["corners_total_mean"])),
            prior_strength_total=float(inputs.get("corners_strength", self.prior["corners_total_strength"])),
            inputs=inputs,
            model_suffix="corners",
        )

    # ---------- Referee factor for cards ----------

    def _league_referee_factor(self, league_id: int, season: int, inputs: Dict[str, Any]) -> Tuple[float, List[str]]:
        notes: List[str] = []
        last_m = int(inputs.get("referee_last_m", 60))
        last_m = max(20, min(last_m, 80))

        min_games = int(inputs.get("referee_min_games", 12))
        base_mean = float(inputs.get("cards_prior_mean", self.prior["cards_total_mean"]))
        min_factor = float(inputs.get("referee_min_factor", 0.75))
        max_factor = float(inputs.get("referee_max_factor", 1.35))

        cache_key = f"ref:{league_id}:{season}:{last_m}:{base_mean}"

        def _compute():
            data = self.client.get("fixtures", {"league": league_id, "season": season, "last": last_m})
            resp = self._response_list(data)

            total_cards = 0.0
            games = 0

            for f in resp:
                fid = (f.get("fixture") or {}).get("id")
                if not fid:
                    continue
                try:
                    stats = self._fixture_stats(int(fid))
                except Exception:
                    continue
                if len(stats) < 2:
                    continue

                c = 0.0
                for team_block in stats[:2]:
                    s_list = team_block.get("statistics") or []
                    for s in s_list:
                        if (s.get("type") or "").lower() == "yellow cards":
                            v = s.get("value")
                            if v is None:
                                v = 0
                            try:
                                c += float(v)
                            except Exception:
                                c += 0.0

                total_cards += c
                games += 1

            if games < min_games:
                return {"factor": 1.0, "games": games, "avg": base_mean}

            avg = total_cards / games
            factor = avg / base_mean if base_mean > 0 else 1.0
            factor = max(min_factor, min(max_factor, factor))
            return {"factor": factor, "games": games, "avg": avg}

        out = self.referee_cache.get_or_set(cache_key, _compute)
        factor = float(out.get("factor", 1.0))
        notes.append(f"League referee(cards) factor={factor:.2f} (avg={out.get('avg')}, games={out.get('games')}).")
        return factor, notes

    def prob_cards_ou(self, fixture_id: int, league_slug: str, selection: str, line: float, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        p, model_name, notes = self.prob_fixture_stats_ou(
            fixture_id, league_slug, selection, line,
            stat_type_name="Yellow Cards",
            prior_mean_total=float(inputs.get("default_lambda", self.prior["cards_total_mean"])),
            prior_strength_total=float(inputs.get("cards_strength", self.prior["cards_total_strength"])),
            inputs=inputs,
            model_suffix="cards",
        )

        use_ref = bool(inputs.get("use_referee_factor", True))
        if not use_ref:
            return p, model_name, notes

        league_id = self._league_id(league_slug)
        season = infer_season()
        factor, n2 = self._league_referee_factor(league_id, season, inputs)
        notes += n2

        # recompute λ once (kept simple, but not duplicated twice)
        fixture = self._fixture(fixture_id)
        home_id, away_id = self._teams_from_fixture(fixture)

        last_n = int(inputs.get("last_n", 8))
        last_n = max(3, min(last_n, 10))

        home_mean, home_n = self._avg_fixture_stat_for_team(home_id, league_id, season, last_n, "Yellow Cards")
        away_mean, away_n = self._avg_fixture_stat_for_team(away_id, league_id, season, last_n, "Yellow Cards")
        lam_obs = (home_mean + away_mean)
        n_proxy = float(min(home_n, away_n)) if (home_n > 0 and away_n > 0) else float(max(home_n, away_n))

        base_mean = float(inputs.get("default_lambda", self.prior["cards_total_mean"]))
        base_strength = float(inputs.get("cards_strength", self.prior["cards_total_strength"]))
        lam = shrink_mean(lam_obs if lam_obs > 0 else base_mean, n_proxy, base_mean, base_strength)
        lam_adj = lam * factor

        alpha = float(inputs.get("dispersion", 10.0))
        sel = normalize_selection(selection)
        if sel == "over":
            p2 = prob_over_nb(line, lam_adj, alpha)
        elif sel == "under":
            p2 = prob_under_nb(line, lam_adj, alpha)
        else:
            raise HTTPException(status_code=400, detail="Invalid selection for cards_ou (over/under)")

        notes.append(f"Cards λ adjusted: {lam:.2f} -> {lam_adj:.2f} (×{factor:.2f}).")
        return clamp01(p2), "football_cards_ou_nb_bayes_referee_league", notes

    def prob_offsides_ou(self, fixture_id: int, league_slug: str, selection: str, line: float, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        return self.prob_fixture_stats_ou(
            fixture_id, league_slug, selection, line,
            stat_type_name="Offsides",
            prior_mean_total=float(inputs.get("default_lambda", self.prior["offsides_total_mean"])),
            prior_strength_total=float(inputs.get("offsides_strength", self.prior["offsides_total_strength"])),
            inputs=inputs,
            model_suffix="offsides",
        )

    # ---------- Team shots / Team SOT (now truly per team, requires team_side) ----------

    def prob_team_shots_ou(self, fixture_id: int, league_slug: str, selection: str, line: float, team_side: str, on_target_only: bool, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        stat_name = "Shots on Goal" if on_target_only else "Total Shots"
        prior_mean = float(inputs.get("default_lambda", self.prior["sot_total_mean"] if on_target_only else self.prior["shots_total_mean"]))
        prior_strength = float(inputs.get("shots_strength", self.prior["sot_total_strength"] if on_target_only else self.prior["shots_total_strength"]))
        suffix = "team_sot" if on_target_only else "team_shots"

        return self.prob_team_stat_ou(
            fixture_id=fixture_id,
            league_slug=league_slug,
            selection=selection,
            line=line,
            team_side=team_side,
            stat_type_name=stat_name,
            prior_mean_team=prior_mean,
            prior_strength_team=prior_strength,
            inputs=inputs,
            model_suffix=suffix,
        )

    # ---------- Player props ----------

    def prob_player_shots_ou(self, league_slug: str, selection: str, line: float, player_id: int, on_target_only: bool, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        league_id = self._league_id(league_slug)
        season = infer_season()
        notes: List[str] = []

        data = self._player_season_stats(league_id, season, player_id)
        stats_list = data.get("statistics") if isinstance(data, dict) else None

        per90_obs = None
        n_proxy = 0.0

        if isinstance(stats_list, list) and stats_list:
            s0 = stats_list[0]
            games = s0.get("games") or {}
            shots = s0.get("shots") or {}

            apps = self._safe_float(games.get("appearences")) or 0.0
            minutes = self._safe_float(games.get("minutes")) or 0.0

            total_shots = self._safe_float(shots.get("total")) or 0.0
            total_sot = self._safe_float(shots.get("on")) or 0.0

            n_proxy = apps if apps > 0 else (minutes / 90.0 if minutes > 0 else 0.0)

            if minutes and minutes > 0:
                factor = 90.0 / minutes
                sh90 = total_shots * factor
                sot90 = total_sot * factor
            elif apps and apps > 0:
                sh90 = total_shots / apps
                sot90 = total_sot / apps
            else:
                sh90 = sot90 = None

            per90_obs = sot90 if on_target_only else sh90

        if on_target_only:
            prior_mean = float(inputs.get("player_sot90_mean", self.prior["player_sot90_mean"]))
            prior_strength = float(inputs.get("player_sot90_strength", self.prior["player_sot90_strength"]))
        else:
            prior_mean = float(inputs.get("player_shots90_mean", self.prior["player_shots90_mean"]))
            prior_strength = float(inputs.get("player_shots90_strength", self.prior["player_shots90_strength"]))

        if per90_obs is None:
            per90_mu = prior_mean
            notes.append("No player stats; using player prior.")
        else:
            per90_mu = shrink_mean(float(per90_obs), float(n_proxy), prior_mean, prior_strength)
            notes.append(f"Player per90 Bayes: obs={per90_obs:.2f} (n~{n_proxy:.1f}) -> {per90_mu:.2f}.")

        expected_minutes = float(inputs.get("expected_minutes", 90.0))
        minutes_factor = max(0.25, min(expected_minutes / 90.0, 1.35))
        lam = per90_mu * minutes_factor
        notes.append(f"Adjusted by expected_minutes={expected_minutes:.0f} => λ={lam:.2f}.")

        sel = normalize_selection(selection)
        if sel == "over":
            p = prob_over_poisson(line, lam)
        elif sel == "under":
            p = prob_under_poisson(line, lam)
        else:
            raise HTTPException(status_code=400, detail="Invalid selection for player shots OU (over/under)")

        model_name = "football_player_sot_ou_bayes" if on_target_only else "football_player_shots_ou_bayes"
        return clamp01(p), model_name, notes

    def prob_player_goal(self, league_slug: str, selection: str, player_id: int, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        league_id = self._league_id(league_slug)
        season = infer_season()
        notes: List[str] = []

        prior_mean = float(inputs.get("player_goal_mean", self.prior["player_goal_mean"]))
        prior_strength = float(inputs.get("player_goal_strength", self.prior["player_goal_strength"]))

        data = self._player_season_stats(league_id, season, player_id)
        stats_list = data.get("statistics") if isinstance(data, dict) else None

        rate_obs = None
        n_proxy = 0.0

        if isinstance(stats_list, list) and stats_list:
            s0 = stats_list[0]
            games = s0.get("games") or {}
            goals = s0.get("goals") or {}

            apps = self._safe_float(games.get("appearences")) or 0.0
            minutes = self._safe_float(games.get("minutes")) or 0.0
            goals_total = self._safe_float(goals.get("total")) or 0.0

            n_proxy = apps if apps > 0 else (minutes / 90.0 if minutes > 0 else 0.0)
            if apps > 0:
                rate_obs = goals_total / apps

        if rate_obs is None:
            rate_mu = prior_mean
            notes.append("No goal stats; using prior.")
        else:
            rate_mu = shrink_mean(float(rate_obs), float(n_proxy), prior_mean, prior_strength)
            notes.append(f"Player goal rate Bayes: obs={rate_obs:.2f} (n~{n_proxy:.1f}) -> {rate_mu:.2f}.")

        lam = rate_mu
        p_yes = 1.0 - math.exp(-lam)

        sel = normalize_selection(selection)
        if sel == "yes":
            return clamp01(p_yes), "football_player_goal_anytime_bayes", notes
        if sel == "no":
            return clamp01(1.0 - p_yes), "football_player_goal_anytime_bayes", notes
        raise HTTPException(status_code=400, detail="Invalid selection for player_goal (yes/no)")

    def prob_player_assist(self, league_slug: str, selection: str, player_id: int, inputs: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        league_id = self._league_id(league_slug)
        season = infer_season()
        notes: List[str] = []

        prior_mean = float(inputs.get("player_assist_mean", self.prior["player_assist_mean"]))
        prior_strength = float(inputs.get("player_assist_strength", self.prior["player_assist_strength"]))

        data = self._player_season_stats(league_id, season, player_id)
        stats_list = data.get("statistics") if isinstance(data, dict) else None

        rate_obs = None
        n_proxy = 0.0

        if isinstance(stats_list, list) and stats_list:
            s0 = stats_list[0]
            games = s0.get("games") or {}
            goals = s0.get("goals") or {}

            apps = self._safe_float(games.get("appearences")) or 0.0
            minutes = self._safe_float(games.get("minutes")) or 0.0
            assists_total = self._safe_float(goals.get("assists")) or 0.0

            n_proxy = apps if apps > 0 else (minutes / 90.0 if minutes > 0 else 0.0)
            if apps > 0:
                rate_obs = assists_total / apps

        if rate_obs is None:
            rate_mu = prior_mean
            notes.append("No assist stats; using prior.")
        else:
            rate_mu = shrink_mean(float(rate_obs), float(n_proxy), prior_mean, prior_strength)
            notes.append(f"Player assist rate Bayes: obs={rate_obs:.2f} (n~{n_proxy:.1f}) -> {rate_mu:.2f}.")

        lam = rate_mu
        p_yes = 1.0 - math.exp(-lam)

        sel = normalize_selection(selection)
        if sel == "yes":
            return clamp01(p_yes), "football_player_assist_anytime_bayes", notes
        if sel == "no":
            return clamp01(1.0 - p_yes), "football_player_assist_anytime_bayes", notes
        raise HTTPException(status_code=400, detail="Invalid selection for player_assist (yes/no)")


# ============================================================
# FASTAPI – REQUEST/RESPONSE MODELS
# ============================================================

SUPPORTED_SPORTS = {"football"}

SUPPORTED_MARKETS = {
    "match_winner_1x2",
    "goals_ou",
    "btts",
    "team_goals_ou",
    "corners_ou",
    "cards_ou",
    "offsides_ou",
    "team_shots_ou",   # now per team (requires team_side)
    "team_sot_ou",     # now per team (requires team_side)
    "player_shots_ou",
    "player_sot_ou",
    "player_goal",
    "player_assist",
}

# Loveable UI metadata: exact required fields + allowed selections
MARKET_META: Dict[str, Dict[str, Any]] = {
    "match_winner_1x2": {
        "required": ["fixture_id"],
        "optional": [],
        "selection_enum": ["home", "draw", "away"],
        "needs_line": False,
        "needs_team_side": False,
        "needs_player_id": False,
    },
    "goals_ou": {
        "required": ["fixture_id", "line"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": False,
        "needs_player_id": False,
    },
    "btts": {
        "required": ["fixture_id"],
        "optional": [],
        "selection_enum": ["yes", "no"],
        "needs_line": False,
        "needs_team_side": False,
        "needs_player_id": False,
    },
    "team_goals_ou": {
        "required": ["fixture_id", "line", "team_side"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": True,
        "needs_player_id": False,
    },
    "corners_ou": {
        "required": ["fixture_id", "line"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": False,
        "needs_player_id": False,
    },
    "cards_ou": {
        "required": ["fixture_id", "line"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": False,
        "needs_player_id": False,
    },
    "offsides_ou": {
        "required": ["fixture_id", "line"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": False,
        "needs_player_id": False,
    },
    "team_shots_ou": {
        "required": ["fixture_id", "line", "team_side"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": True,
        "needs_player_id": False,
    },
    "team_sot_ou": {
        "required": ["fixture_id", "line", "team_side"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": True,
        "needs_player_id": False,
    },
    "player_shots_ou": {
        "required": ["player_id", "line"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": False,
        "needs_player_id": True,
    },
    "player_sot_ou": {
        "required": ["player_id", "line"],
        "optional": [],
        "selection_enum": ["over", "under"],
        "needs_line": True,
        "needs_team_side": False,
        "needs_player_id": True,
    },
    "player_goal": {
        "required": ["player_id"],
        "optional": [],
        "selection_enum": ["yes", "no"],
        "needs_line": False,
        "needs_team_side": False,
        "needs_player_id": True,
    },
    "player_assist": {
        "required": ["player_id"],
        "optional": [],
        "selection_enum": ["yes", "no"],
        "needs_line": False,
        "needs_team_side": False,
        "needs_player_id": True,
    },
}

class EVRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    sport: str = Field(..., description="football")
    league: str = Field(..., description="league slug, e.g. premier_league")
    market: str = Field(..., description="see SUPPORTED_MARKETS")
    selection: str = Field(..., description="varies by market")
    odds: float = Field(..., gt=1.0, description="decimal odds")

    fixture_id: Optional[int] = Field(None, description="Fixture id for match markets")
    player_id: Optional[int] = Field(None, description="Player id for player markets")
    team_side: Optional[str] = Field(None, description="'home' or 'away' for team markets")
    line: Optional[float] = Field(None, description="O/U line")

    inputs: Dict[str, Any] = Field(default_factory=dict, description="extra model parameters")

class EVResponse(BaseModel):
    probability: float
    implied_probability: float
    edge: float
    ev: float
    edge_percent: float
    model: str
    notes: List[str]


# ============================================================
# FASTAPI – APP
# ============================================================

app = FastAPI(title="EV Engine API", version="1.0.0")

# CORS for Loveable (set LOVEABLE_ORIGINS="https://your-loveable.app,https://*.loveable.app" etc)
origins_env = os.getenv("LOVEABLE_ORIGINS", "*")
if origins_env.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily initialized globals to avoid Render import crash
client: Optional[ApiFootballClient] = None
football_model: Optional[FootballModel] = None
init_error: Optional[str] = None

@app.on_event("startup")
def startup_init():
    global client, football_model, init_error
    try:
        API_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
        client = ApiFootballClient(ApiFootballConfig(base_url=API_BASE_URL))
        football_model = FootballModel(client, load_leagues())
        init_error = None
    except Exception as e:
        # Keep API alive for /health and /meta; /ev will return 503 with this detail
        init_error = str(e)
        client = None
        football_model = None

@app.get("/health")
def health():
    return {
        "status": "ok" if init_error is None else "degraded",
        "init_error": init_error,
    }

@app.get("/meta")
def meta():
    return {
        "sports": sorted(list(SUPPORTED_SPORTS)),
        "markets": sorted(list(SUPPORTED_MARKETS)),
        "market_meta": MARKET_META,  # Loveable can build UI from this
        "season": infer_season(),
        "leagues": (football_model.leagues if football_model else load_leagues()),
        "cache": {
            "api_cache_ttl_seconds": (client.cfg.cache_ttl_seconds if client else None),
            "api_cache_max_items": (client.cfg.cache_max_items if client else None),
        },
        "selection_aliases": sorted(list(set(_SELECTION_ALIASES.keys()))),
    }

def require_fields(req: EVRequest, fields: List[str]) -> None:
    missing = [f for f in fields if getattr(req, f) is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")

@app.post("/ev", response_model=EVResponse)
def calculate_ev_endpoint(req: EVRequest):
    if init_error is not None or football_model is None:
        raise HTTPException(status_code=503, detail=f"Service not initialized: {init_error or 'unknown error'}")

    sport = (req.sport or "").lower().strip()
    league = (req.league or "").lower().strip()
    market = (req.market or "").lower().strip()

    if sport not in SUPPORTED_SPORTS:
        raise HTTPException(status_code=400, detail=f"Unsupported sport: {sport}")
    if market not in SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Unsupported market: {market}")

    # Normalize selection & team_side for Loveable UX
    req.selection = normalize_selection(req.selection)
    req.team_side = normalize_team_side(req.team_side)

    # Validate selection enum (Loveable-friendly error)
    meta = MARKET_META.get(market)
    if meta:
        allowed = meta.get("selection_enum") or []
        if allowed and req.selection not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid selection '{req.selection}'. Allowed: {allowed}")

    p: Optional[float] = None
    model_name = "unknown"
    notes: List[str] = []

    if market == "match_winner_1x2":
        require_fields(req, ["fixture_id"])
        p, model_name, notes = football_model.prob_1x2(req.fixture_id, league, req.selection, req.inputs)

    elif market == "goals_ou":
        require_fields(req, ["fixture_id", "line"])
        p, model_name, notes = football_model.prob_goals_ou(req.fixture_id, league, req.selection, req.line, req.inputs)

    elif market == "btts":
        require_fields(req, ["fixture_id"])
        p, model_name, notes = football_model.prob_btts(req.fixture_id, league, req.selection, req.inputs)

    elif market == "team_goals_ou":
        require_fields(req, ["fixture_id", "line", "team_side"])
        p, model_name, notes = football_model.prob_team_goals_ou(req.fixture_id, league, req.selection, req.line, req.team_side, req.inputs)

    elif market == "corners_ou":
        require_fields(req, ["fixture_id", "line"])
        p, model_name, notes = football_model.prob_corners_ou(req.fixture_id, league, req.selection, req.line, req.inputs)

    elif market == "cards_ou":
        require_fields(req, ["fixture_id", "line"])
        p, model_name, notes = football_model.prob_cards_ou(req.fixture_id, league, req.selection, req.line, req.inputs)

    elif market == "offsides_ou":
        require_fields(req, ["fixture_id", "line"])
        p, model_name, notes = football_model.prob_offsides_ou(req.fixture_id, league, req.selection, req.line, req.inputs)

    elif market == "team_shots_ou":
        require_fields(req, ["fixture_id", "line", "team_side"])
        p, model_name, notes = football_model.prob_team_shots_ou(req.fixture_id, league, req.selection, req.line, req.team_side, False, req.inputs)

    elif market == "team_sot_ou":
        require_fields(req, ["fixture_id", "line", "team_side"])
        p, model_name, notes = football_model.prob_team_shots_ou(req.fixture_id, league, req.selection, req.line, req.team_side, True, req.inputs)

    elif market == "player_shots_ou":
        require_fields(req, ["player_id", "line"])
        p, model_name, notes = football_model.prob_player_shots_ou(league, req.selection, req.line, req.player_id, False, req.inputs)

    elif market == "player_sot_ou":
        require_fields(req, ["player_id", "line"])
        p, model_name, notes = football_model.prob_player_shots_ou(league, req.selection, req.line, req.player_id, True, req.inputs)

    elif market == "player_goal":
        require_fields(req, ["player_id"])
        p, model_name, notes = football_model.prob_player_goal(league, req.selection, req.player_id, req.inputs)

    elif market == "player_assist":
        require_fields(req, ["player_id"])
        p, model_name, notes = football_model.prob_player_assist(league, req.selection, req.player_id, req.inputs)

    if p is None:
        raise HTTPException(status_code=500, detail="Probability calculation failed")

    p = clamp01(p)
    ev_data = calculate_ev(p, req.odds)

    return EVResponse(
        probability=ev_data["probability"],
        implied_probability=ev_data["implied_probability"],
        edge=ev_data["edge"],
        ev=ev_data["ev"],
        edge_percent=ev_data["edge_percent"],
        model=model_name,
        notes=notes,
    )


"""
RUN LOCAL:
  pip install fastapi uvicorn[standard] requests
  export API_FOOTBALL_KEY="YOUR_KEY"
  uvicorn main:app --reload

DEPLOY RENDER:
  - main.py (this file)
  - requirements.txt:
      fastapi
      uvicorn[standard]
      requests
  - Start command:
      uvicorn main:app --host 0.0.0.0 --port $PORT
  - Env vars:
      API_FOOTBALL_KEY=...
      (optional) API_FOOTBALL_BASE_URL=https://v3.football.api-sports.io
      (optional) FOOTBALL_SEASON=2025
      (optional) FOOTBALL_LEAGUES_JSON={"premier_league":39,...}
      (optional) LOVEABLE_ORIGINS=*
        or comma-separated origins
"""