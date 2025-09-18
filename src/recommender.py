# FAISS retrieval + hybrid reranking + explanations + tracing (robusto a nombres de columnas)
import os, re, json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).as_posix()

# ---------- Hiper-parámetros ----------
ALPHA = 0.55
W_BUDGET, W_TYPE, W_ACTIV, W_SEASON = 0.32, 0.28, 0.30, 0.10
BUDGET_MARGIN, BUDGET_BAND = 10, 60
TOP_K_RETRIEVE = 200
SEMANTIC_BOOST_TOPN = 10
SEMANTIC_BOOST_EPS  = 0.04
SEMANTIC_BOOST_THR  = 0.62

SEASON_NEIGHBORS = {
    "spring":{"summer"},
    "summer":{"spring","autumn"},
    "autumn":{"summer","winter"},
    "winter":{"autumn"},
}

# ---------- Diccionarios ----------
TYPE_KEYWORDS = {
    "coastal":  [r"\b(coastal|seaside|oceanfront|waterfront|shore|bay|harbor|harbour)\b"],
    "city":     [r"\b(city|urban|downtown|nightlife|shopping)\b"],
    "nature":   [r"\b(nature|mountain|mountains|national park|forest|lake|lakes|hiking|trail|outdoors)\b"],
    "culture":  [r"\b(culture|cultural|history|historical|heritage|museum|museums|art|architecture|old town|cathedral|castle)\b"],
}

ACTIVITY_SYNONYMS = {
    "beach":       [r"\b(beach|sunbath|sunbathing|swim|swimming)\b"],
    "seafood":     [r"\b(seafood|fish market|fresh fish|oysters?|shellfish|fish restaurant|sea[- ]?to[- ]?table|ceviche|grilled fish)\b"],
    "museums":     [r"\b(museum|museums|gallery|galleries|exhibit|art)\b"],
    "hiking":      [r"\b(hiking|trek|trekking|trail|trails)\b"],
    "nightlife":   [r"\b(nightlife|bar|bars|club|clubs|party|parties)\b"],
    "food":        [r"\b(food|cuisine|restaurant|restaurants|street food|gastronomy)\b"],
    "architecture":[r"\b(architecture|architectural|old town|cathedral|castle|historic)\b"],
    "shopping":    [r"\b(shopping|market|markets|bazaar|mall|malls|boutique|boutiques)\b"],
    "wine":        [r"\b(wine|winery|wineries|vineyard|vineyards)\b"],
    "surfing":     [r"\b(surf|surfing)\b"],
    "diving":      [r"\b(?:snorkel(?:ing)?|div(?:e|ing)|scuba|reef|coral)\b"],  # sin grupos capturados
}

# ---------- Carga de artefactos ----------
def load_artifacts(data_dir=DATA_DIR):
    index = faiss.read_index(os.path.join(data_dir, "faiss.index"))
    id_map = np.load(os.path.join(data_dir, "id_map.npy"))
    with open(os.path.join(data_dir, "index_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    df_rows = pd.read_pickle(os.path.join(data_dir, "rows.pkl"))
    model = SentenceTransformer(meta["model"])
    return index, id_map, df_rows, model, meta

INDEX, IDMAP, DF_ROWS, MODEL, META = load_artifacts()

# ---------- Normalización de nombres de columnas ----------
# Sinónimos por columna canónica
_COLUMN_SYNONYMS = {
    "City":       ["city", "ciudad"],
    "Country":    ["country", "pais", "país"],
    "Continent":  ["continent", "continente", "region", "región"],
    "Budget":     ["budget", "presupuesto", "price_range"],
    "Season":     ["season", "estacion", "estación"],
    "Tags":       ["tags", "etiquetas"],
    "Activities": ["activities", "actividads", "actividades"],
    "Style":      ["style", "estilo"],
    "FullText":   ["fulltext", "text", "texto", "doc", "document"],
    "is_coastal": ["is_coastal", "coastal", "costa", "iscoastal"],
    "Activity 1": ["activity 1", "activity1"],
    "Activity 2": ["activity 2", "activity2"],
}

def _build_colmap(df: pd.DataFrame):
    # índice de columnas normalizado
    norm = {c.lower().strip(): c for c in df.columns}
    colmap = {}
    for canonical, candidates in _COLUMN_SYNONYMS.items():
        found = None
        for cand in [canonical] + candidates:
            key = cand.lower().strip()
            if key in norm:
                found = norm[key]; break
        colmap[canonical] = found  # puede quedar None
    return colmap

COLS = _build_colmap(DF_ROWS)

def _has(col):  # ¿existe columna canónica?
    return COLS.get(col) is not None

def _col(col):  # nombre real de la columna (o None)
    return COLS.get(col)

def _series(df, col, default=None):
    c = _col(col)
    if c is None:
        if default is None:
            return pd.Series([], dtype=object)
        return pd.Series([default]*len(df), index=df.index)
    return df[c]

# ---------- Utils de puntuación ----------
def parse_budget_range(budget_str):
    s = str(budget_str).lower().replace("—","-").replace("–","-").replace("−","-")
    m = re.match(r"^\s*(\d{1,5})\s*-\s*(\d{1,5})\s*usd\s*$", s)
    if not m: return None, None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi: lo, hi = hi, lo
    return lo, hi

def avg_budget(budget_str, default=9999):
    lo, hi = parse_budget_range(budget_str)
    return (lo + hi) / 2 if lo is not None else default

def budget_score(budget_str, cap, margin=BUDGET_MARGIN, band=BUDGET_BAND):
    if cap is None: return 0.5
    b = avg_budget(budget_str)
    thresh = cap + margin
    if b <= thresh: return 1.0
    return max(0.0, 1.0 - (b - thresh)/band)

def season_score(item_season, desired):
    if not desired: return 0.5
    s = str(item_season).lower().strip()
    if s == desired: return 1.0
    if desired in SEASON_NEIGHBORS and s in SEASON_NEIGHBORS[desired]: return 0.5
    return 0.0

def jaccard(wants, haves):
    if not wants: return 0.5
    if not haves: return 0.0
    inter = len(wants & haves); uni = len(wants | haves)
    return 0.0 if uni==0 else inter/uni

def _safe_col(row, canonical):
    c = _col(canonical)
    return "" if c is None else str(row.get(c, ""))

def _blob_from_row(row):
    return " ".join([
        _safe_col(row, "Tags"),
        _safe_col(row, "Activities"),
        _safe_col(row, "Activity 1"),
        _safe_col(row, "Activity 2"),
        _safe_col(row, "Style"),
        _safe_col(row, "FullText"),
    ]).lower()

def item_types(row):
    blob = _blob_from_row(row)
    types = set()
    for t, pats in TYPE_KEYWORDS.items():
        if any(re.search(p, blob) for p in pats): types.add(t)
    # Señal explícita manda si existe
    if _col("is_coastal") in row and row[_col("is_coastal")] is True: types.add("coastal")
    if _col("is_coastal") in row and row[_col("is_coastal")] is False and "coastal" in types: types.discard("coastal")
    return types

def item_activities(row):
    blob = _blob_from_row(row)
    acts = set()
    # 'beach' solo si costa explícita
    if bool(row.get(_col("is_coastal"), False)) if _has("is_coastal") else False:
        if any(re.search(p, blob) for p in ACTIVITY_SYNONYMS["beach"]): acts.add("beach")
    for a, pats in ACTIVITY_SYNONYMS.items():
        if a == "beach": continue
        if any(re.search(p, blob) for p in pats): acts.add(a)
    return acts

# --- Continente desde prompt (robusto a variantes) ---
CONTINENTS = {
    # básicos
    "europe": "europe", "european": "europe",
    "asia": "asia", "asian": "asia",
    "africa": "africa", "african": "africa",
    "north america": "north america", "north-america": "north america",
    "south america": "south america", "south-america": "south america",
    "latin america": "south america", "latin-america": "south america",
    "oceania": "oceania", "australasia": "oceania",
    # abreviaturas frecuentes
    "latam": "south america",
}

def _continent_key_to_regex(key: str) -> str:
    """
    Convierte 'south america' -> r'\bsouth[ -]america\b'
    y 'latin-america' -> r'\blatin[ -]america\b'
    """
    parts = re.split(r"[ -]+", key.strip())
    if len(parts) == 1:
        return rf"\b{re.escape(parts[0])}\b"
    joined = r"[ -]".join(map(re.escape, parts))
    return rf"\b{joined}\b"

# precompila patrones
_CONT_PATTERNS = [(re.compile(_continent_key_to_regex(k)), v) for k, v in CONTINENTS.items()]

def detect_continent(t: str):
    text = (t or "").lower()
    for pat, v in _CONT_PATTERNS:
        if pat.search(text):
            return v
    return None

def extract_preferences(text):
    t = (text or "").lower()

    # season
    season = None
    for s in ["spring","summer","autumn","winter","fall"]:
        if re.search(rf"\b{s}\b", t):
            season = "autumn" if s=="fall" else s
            break

    # budget
    m = re.search(r"(?:under|less than|<=|≤)\s*(\d{2,4})\s*(?:usd|dollars|\$)?", t)
    if not m:
        m = re.search(r"\b(?:~|around|about)\s*(\d{2,4})\s*(?:usd|dollars|\$)?", t)
    budget_cap = int(m.group(1)) if m else None
    if budget_cap is None and any(k in t for k in ["affordable","cheap","budget"]):
        budget_cap = 100

    # types
    type_wants = set()
    for typ in ["coastal","city","nature","culture"]:
        if any(re.search(p, t) for p in TYPE_KEYWORDS[typ]): type_wants.add(typ)

    # Sugerir costa si el prompt habla de islas
    if re.search(r"\bisland(s)?\b", t):
        type_wants.add("coastal")

    # Excluir costa si el prompt lo indica
    coastal_forbid = bool(re.search(r"\b(no\s+coast|not\s+coastal|away\s+from\s+coast|inland\s+only)\b", t))

    # activities
    act_wants = set()
    for a, pats in ACTIVITY_SYNONYMS.items():
        if any(re.search(p, t) for p in pats): act_wants.add(a)

    continent = detect_continent(t)

    return {
        "season": season,
        "budget_cap": budget_cap,
        "type_wants": type_wants,
        "act_wants": act_wants,
        "continent": continent,
        "coastal_forbid": coastal_forbid,
    }

def make_query_text(prefs, user_text):
    p = prefs
    parts = [f"QUERY: TYPE={','.join(sorted(p['type_wants'])) or 'any'}; "
             f"SEASON={p['season'] or 'any'}; "
             f"ACTIVITIES={','.join(sorted(p['act_wants'])) or 'any'}; "
             f"BUDGET<={p['budget_cap'] if p['budget_cap'] else 'unspecified'} USD; "
             f"CONTINENT={p['continent'] or 'any'}; "
             f"COASTAL_FORBID={p['coastal_forbid']}."]
    parts.append(user_text)
    return "\n".join(parts)

# ---------- Prefiltros ----------
def _apply_coastal_filter(df: pd.DataFrame, must: bool, forbid: bool) -> pd.DataFrame:
    sub = df
    if must:
        if _has("is_coastal"):
            sub = sub[sub[_col("is_coastal")] == True]
        else:
            tag_ok = _series(sub, "Tags").astype(str).str.contains(
                r"\b(coastal|seaside|oceanfront|waterfront|shore|bay|harbor|harbour)\b",
                case=False, na=False
            )
            sty_ok = _series(sub, "Style").astype(str).str.contains(
                r"\b(coastal|seaside|oceanfront|waterfront)\b",
                case=False, na=False
            )
            sub = sub[tag_ok | sty_ok]
    if forbid:
        if _has("is_coastal"):
            sub = sub[sub[_col("is_coastal")] == False]
        else:
            tag_coast = _series(sub, "Tags").astype(str).str.contains(
                r"\b(coastal|seaside|oceanfront|waterfront|shore|bay|harbor|harbour)\b",
                case=False, na=False
            )
            sty_coast = _series(sub, "Style").astype(str).str.contains(
                r"\b(coastal|seaside|oceanfront|waterfront)\b",
                case=False, na=False
            )
            sub = sub[~(tag_coast | sty_coast)]
    return sub

def _prefilter_season(sub: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    desired = prefs.get("season")
    if not desired or not _has("Season"): return sub
    s = _series(sub, "Season").astype(str).str.lower().str.strip()
    exact = sub[s == desired]
    if not exact.empty: return exact
    near = SEASON_NEIGHBORS.get(desired, set())
    return sub[s.isin(near)] if near else sub

def prefilter_candidates(df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    sub = df

    # 1) CONTINENTE (candado estricto)
    cont = prefs.get("continent")
    if cont and _has("Continent"):
        scont = _series(sub, "Continent").astype(str).str.lower().str.strip()
        sub = sub[scont == cont]
    # guarda subset por continente para fallback controlado
    base_cont_sub = sub.copy()

    # 2) COASTAL: must/forbid
    wants = prefs.get("type_wants", set()) or set()
    must_coastal = "coastal" in wants
    forbid_coastal = bool(prefs.get("coastal_forbid", False))
    if must_coastal and forbid_coastal:  # conflicto -> gana forbid
        must_coastal = False
    if must_coastal or forbid_coastal:
        sub = _apply_coastal_filter(sub, must=must_coastal, forbid=forbid_coastal)

    # 3) ACTIVIDADES ACUÁTICAS (exige costa + keywords)
    act_wants = prefs.get("act_wants", set()) or set()
    if {"diving","surfing"} & act_wants:
        sub = _apply_coastal_filter(sub, must=True, forbid=False)
        water_pat = r"(?:snorkel(?:ing)?|div(?:e|ing)|scuba|reef|coral|surf(?:ing)?)"
        act_ok = _series(sub, "Activities").astype(str).str.contains(water_pat, case=False, na=False)
        tag_ok = _series(sub, "Tags").astype(str).str.contains(water_pat, case=False, na=False)
        sub = sub[act_ok | tag_ok]

    # 4) SEASON (prefer exact; si vacío, vecinos)
    sub = _prefilter_season(sub, prefs)

    # 5) Fallbacks defensivos: NUNCA soltar continente si venía en el prompt
    if sub.empty:
        if len(base_cont_sub) > 0:
            # relajamos coastal/acuáticas/season pero mantenemos el continente
            sub = base_cont_sub
        else:
            # si no existía nada en ese continente en el CSV, devolvemos todo
            sub = df

    return sub


# ---------- Recommend ----------
def recommend(user_text, top_k=10, alpha=ALPHA):
    prefs = extract_preferences(user_text)
    qtext = make_query_text(prefs, user_text)

    # 1) Prefiltro DURO
    df_pref = prefilter_candidates(DF_ROWS, prefs)

    # Si vacío, relajamos preservando continente si venía
    if df_pref.empty:
        cont = prefs.get("continent")
        if cont and _has("Continent"):
            scont = _series(DF_ROWS, "Continent").astype(str).str.lower().str.strip()
            df_pref = DF_ROWS[scont == cont]
            if df_pref.empty:
                df_pref = DF_ROWS
        else:
            df_pref = DF_ROWS

    # 2) Recuperación FAISS global + cruce
    q = MODEL.encode([qtext], normalize_embeddings=True)
    sims, idxs = INDEX.search(np.asarray(q, dtype="float32"), k=max(TOP_K_RETRIEVE, top_k))
    sims = sims[0]; idxs = idxs[0]
    row_ids = IDMAP[idxs]

    allowed_index = set(df_pref.index.tolist())
    mask = [rid in allowed_index for rid in row_ids]
    if not any(mask) and len(df_pref) > 0:
        # rebúsqueda intrafiltro
        cand_texts = (_series(df_pref, "FullText").astype(str)
                      if _has("FullText") else
                      (_series(df_pref, "City").astype(str) + ", " + _series(df_pref, "Country").astype(str) + " | " +
                       _series(df_pref, "Tags").astype(str) + " | " + _series(df_pref, "Activities").astype(str)))
        emb_cands = MODEL.encode(cand_texts.tolist(), normalize_embeddings=True)
        d = emb_cands.shape[1]
        local_index = faiss.IndexFlatIP(d)
        local_index.add(emb_cands.astype("float32"))
        sims2, idxs2 = local_index.search(np.asarray(q, dtype="float32"),
                                          k=min(len(df_pref), max(TOP_K_RETRIEVE, top_k)))
        sims = sims2[0]
        row_ids = df_pref.index.to_numpy()[idxs2[0]]
        mask = np.ones_like(row_ids, dtype=bool)

    row_ids = np.asarray(row_ids)[mask]
    sims = np.asarray(sims)[mask]

    if len(row_ids) == 0:
        return DF_ROWS.head(0)

    # Candidatos globales por FAISS
    cands = DF_ROWS.iloc[row_ids].copy()

    # Refuerzo final: candado de continente
    cont = prefs.get("continent")
    if cont and _has("Continent"):
        scont_c = _series(cands, "Continent").astype(str).str.lower().str.strip()
        mask = (scont_c == cont).to_numpy()

        if mask.any():
            cands = cands[mask].copy()
            sims = np.asarray(sims)[mask]
        else:
            # No hay intersección con FAISS -> intrabúsqueda solo dentro del continente
            pool = DF_ROWS[_series(DF_ROWS, "Continent").astype(str).str.lower().str.strip() == cont]
            if not pool.empty:
                cand_texts = (_series(pool, "FullText").astype(str)
                            if _has("FullText") else
                            (_series(pool, "City").astype(str) + ", " + _series(pool, "Country").astype(str) + " | " +
                            _series(pool, "Tags").astype(str) + " | " + _series(pool, "Activities").astype(str)))
                emb_pool = MODEL.encode(cand_texts.tolist(), normalize_embeddings=True)
                d = emb_pool.shape[1]
                local_index = faiss.IndexFlatIP(d)
                local_index.add(emb_pool.astype("float32"))
                # q ya existe del encode de la consulta
                sims2, idxs2 = local_index.search(np.asarray(q, dtype="float32"),
                                                k=min(len(pool), max(TOP_K_RETRIEVE, top_k)))
                cands = pool.iloc[idxs2[0]].copy()
                sims = sims2[0]
            else:
                # Último recurso: devolver algo del continente con sim 0.0
                scont_all = _series(DF_ROWS, "Continent").astype(str).str.lower().str.strip()
                cands = DF_ROWS[scont_all == cont].copy().head(top_k)
                sims = np.zeros(len(cands), dtype="float32")

    # Alinear similitudes y continuar
    cands["__sim"] = np.asarray(sims, dtype="float32")
    cands.reset_index(drop=True, inplace=True)


    # 3) Reranking estructurado
    s_type, s_act, s_season, s_budget = [], [], [], []
    for _, r in cands.iterrows():
        it_types = item_types(r)
        it_acts  = item_activities(r)
        s_type.append(jaccard(prefs["type_wants"], it_types))
        s_act.append(jaccard(prefs["act_wants"], it_acts))
        s_season.append(season_score(r.get(_col("Season"),""), prefs["season"]))
        s_budget.append(budget_score(r.get(_col("Budget"),""), prefs["budget_cap"]))

    s_type   = np.array(s_type, dtype="float32")
    s_act    = np.array(s_act, dtype="float32")
    s_season = np.array(s_season, dtype="float32")
    s_budget = np.array(s_budget, dtype="float32")

    structured = (W_BUDGET*s_budget + W_TYPE*s_type + W_ACTIV*s_act + W_SEASON*s_season).astype("float32")
    final = float(alpha) * cands["__sim"].values + (1.0 - float(alpha)) * structured
    cands["score"] = final

    # 4) Semantic boost
    if SEMANTIC_BOOST_TOPN and SEMANTIC_BOOST_EPS > 0 and len(cands) > 0:
        topN = min(SEMANTIC_BOOST_TOPN, len(cands))
        topN_idx = np.argsort(-cands["score"].values)[:topN]
        docs = [
            " | ".join([
                str(cands.iloc[i].get(_col("Activities"),"")),
                str(cands.iloc[i].get(_col("Tags"),"")),
                str(cands.iloc[i].get(_col("Style"),"")),
            ]) for i in topN_idx
        ]
        q_acts_text = "activities: " + (", ".join(sorted(prefs.get("act_wants", []))) or "any")
        emb_q = MODEL.encode([q_acts_text], normalize_embeddings=True)
        emb_d = MODEL.encode(docs,           normalize_embeddings=True)
        sims2 = (emb_q @ emb_d.T).flatten()
        boosts = np.where(
            sims2 > SEMANTIC_BOOST_THR,
            (sims2 - SEMANTIC_BOOST_THR) * SEMANTIC_BOOST_EPS,
            0.0
        ).astype("float32")
        scores = cands["score"].to_numpy(copy=True)
        scores[topN_idx] += boosts
        cands["score"] = scores

    # 5) Salida
    cands["explanation"] = [
        f"sim={cands['__sim'].iloc[i]:.2f} | budget={s_budget[i]:.2f} | type={s_type[i]:.2f} | acts={s_act[i]:.2f} | season={s_season[i]:.2f}"
        for i in range(len(cands))
    ]

    cands = cands.sort_values("score", ascending=False)
    if _has("City") and _has("Country"):
        cands = cands.drop_duplicates(subset=[_col("City"), _col("Country")], keep="first")

    out_cols = ["City","Country","Continent","Budget","Season","Tags","Activities","Style","explanation","score"]
    out_cols_real = [c for c in [_col(x) for x in out_cols] if c is not None] + ["explanation","score"]
    out_cols_real = list(dict.fromkeys(out_cols_real))  # sin duplicados
    return cands[out_cols_real].head(top_k).reset_index(drop=True)

# ===== DEBUGGING / TRACING =====
def _prefilter_candidates_with_trace(df: pd.DataFrame, prefs: dict):
    trace = {
        "prefs": prefs,
        "sizes": {},
        "notes": [],
        "examples": {},
        "colmap": COLS.copy(),
        "has_columns": {k: _has(k) for k in COLS.keys()},
    }
    sub = df
    trace["sizes"]["start"] = len(sub)

    # 1) CONTINENTE
    cont = prefs.get("continent")
    if cont and _has("Continent"):
        scont = _series(sub, "Continent").astype(str).str.lower().str.strip()
        mask_cont = (scont == cont)
        sub = sub[mask_cont]
        trace["sizes"]["after_continent"] = len(sub)
        trace["notes"].append(f"continent='{cont}' applied (strict).")
        if len(sub) == 0:
            trace["examples"]["available_continents"] = (
                _series(df, "Continent").astype(str).str.lower().str.strip().value_counts().head(10).to_dict()
                if _has("Continent") else {}
            )

    # 2) COASTAL must/forbid
    wants = prefs.get("type_wants", set()) or set()
    must_coastal = "coastal" in wants
    forbid_coastal = bool(prefs.get("coastal_forbid", False))
    if must_coastal and forbid_coastal:
        must_coastal = False
        trace["notes"].append("coastal: conflict (must & forbid) -> forbid wins.")
    if must_coastal or forbid_coastal:
        before = len(sub)
        sub = _apply_coastal_filter(sub, must=must_coastal, forbid=forbid_coastal)
        trace["sizes"]["after_coastal"] = len(sub)
        trace["notes"].append(f"coastal filter -> must={must_coastal}, forbid={forbid_coastal} (Δ={before-len(sub)})")

    # 3) ACTIVIDADES ACUÁTICAS
    act_wants = prefs.get("act_wants", set()) or set()
    if {"diving","surfing"} & act_wants:
        before = len(sub)
        sub = _apply_coastal_filter(sub, must=True, forbid=False)
        water_pat = r"(?:snorkel(?:ing)?|div(?:e|ing)|scuba|reef|coral|surf(?:ing)?)"
        act_ok = _series(sub, "Activities").astype(str).str.contains(water_pat, case=False, na=False)
        tag_ok = _series(sub, "Tags").astype(str).str.contains(water_pat, case=False, na=False)
        sub = sub[act_ok | tag_ok]
        trace["sizes"]["after_water_acts"] = len(sub)
        trace["notes"].append(f"water acts filter (diving/surfing) (Δ={before-len(sub)})")

    # 4) SEASON
    before = len(sub)
    sub = _prefilter_season(sub, prefs)
    trace["sizes"]["after_season"] = len(sub)
    if prefs.get("season"):
        trace["notes"].append(f"season='{prefs['season']}' applied (Δ={before-len(sub)})")

    # 5) Fallback preservando continente
    if cont and len(sub) == 0 and _has("Continent"):
        scont = _series(df, "Continent").astype(str).str.lower().str.strip()
        sub = df[scont == cont]
        trace["sizes"]["fallback_preserve_continent"] = len(sub)
        trace["notes"].append("fallback: relax filters but preserve continent.")

    # Ejemplos HEAD (solo las cols que existan)
    head_cols_cand = [c for c in [_col("City"), _col("Country"), _col("Continent"), _col("Budget"), _col("Season")] if c]
    if len(sub) > 0 and head_cols_cand:
        head = sub[head_cols_cand].head(5).to_dict(orient="records")
        trace["examples"]["candidates_head"] = head

    return sub.copy(), trace

def recommend_with_trace(user_text, top_k=10, alpha=ALPHA):
    prefs = extract_preferences(user_text)
    qtext = make_query_text(prefs, user_text)

    df_pref, pf_trace = _prefilter_candidates_with_trace(DF_ROWS, prefs)
    pf_trace["sizes"]["prefilter_result"] = len(df_pref)

    used_local_index = False
    if df_pref.empty and not prefs.get("continent"):
        df_pref = DF_ROWS
        pf_trace["notes"].append("no continent -> prefilter empty -> fallback to global DF_ROWS.")
        pf_trace["sizes"]["prefilter_result_after_global_fallback"] = len(df_pref)

    # FAISS global
    q = MODEL.encode([qtext], normalize_embeddings=True)
    sims, idxs = INDEX.search(np.asarray(q, dtype="float32"), k=max(TOP_K_RETRIEVE, top_k))
    sims = sims[0]; idxs = idxs[0]
    row_ids = IDMAP[idxs]

    allowed_index = set(df_pref.index.tolist())
    mask = np.array([rid in allowed_index for rid in row_ids], dtype=bool)
    pf_trace["sizes"]["faiss_hits"] = int(mask.sum())
    pf_trace["sizes"]["faiss_total"] = len(row_ids)

    if not mask.any() and len(df_pref) > 0:
        used_local_index = True
        cand_texts = (_series(df_pref, "FullText").astype(str)
                      if _has("FullText") else
                      (_series(df_pref, "City").astype(str) + ", " + _series(df_pref, "Country").astype(str) + " | " +
                       _series(df_pref, "Tags").astype(str) + " | " + _series(df_pref, "Activities").astype(str)))
        emb_cands = MODEL.encode(cand_texts.tolist(), normalize_embeddings=True)
        d = emb_cands.shape[1]
        local_index = faiss.IndexFlatIP(d)
        local_index.add(emb_cands.astype("float32"))
        sims2, idxs2 = local_index.search(np.asarray(q, dtype="float32"),
                                          k=min(len(df_pref), max(TOP_K_RETRIEVE, top_k)))
        sims = sims2[0]
        row_ids = df_pref.index.to_numpy()[idxs2[0]]
        mask = np.ones_like(row_ids, dtype=bool)
        pf_trace["notes"].append("no FAISS intersection -> used local intrafilter search.")
        pf_trace["sizes"]["local_hits"] = len(row_ids)

    row_ids = np.asarray(row_ids)[mask]
    sims = np.asarray(sims)[mask]

    if len(row_ids) == 0:
        pf_trace["notes"].append("no candidates after retrieval. returning empty.")
        return DF_ROWS.head(0), pf_trace

    cands = DF_ROWS.iloc[row_ids].copy()
    cands["__sim"] = sims
    cands.reset_index(drop=True, inplace=True)

    s_type, s_act, s_season, s_budget = [], [], [], []
    for _, r in cands.iterrows():
        it_types = item_types(r)
        it_acts  = item_activities(r)
        s_type.append(jaccard(prefs["type_wants"], it_types))
        s_act.append(jaccard(prefs["act_wants"], it_acts))
        s_season.append(season_score(r.get(_col("Season"),""), prefs["season"]))
        s_budget.append(budget_score(r.get(_col("Budget"),""), prefs["budget_cap"]))

    s_type   = np.array(s_type, dtype="float32")
    s_act    = np.array(s_act, dtype="float32")
    s_season = np.array(s_season, dtype="float32")
    s_budget = np.array(s_budget, dtype="float32")

    structured = (W_BUDGET*s_budget + W_TYPE*s_type + W_ACTIV*s_act + W_SEASON*s_season).astype("float32")
    final = float(alpha) * cands["__sim"].values + (1.0 - float(alpha)) * structured
    cands["score"] = final

    # Semantic boost
    if SEMANTIC_BOOST_TOPN and SEMANTIC_BOOST_EPS > 0 and len(cands) > 0:
        topN = min(SEMANTIC_BOOST_TOPN, len(cands))
        topN_idx = np.argsort(-cands["score"].values)[:topN]
        docs = [
            " | ".join([
                str(cands.iloc[i].get(_col("Activities"),"")),
                str(cands.iloc[i].get(_col("Tags"),"")),
                str(cands.iloc[i].get(_col("Style"),"")),
            ]) for i in topN_idx
        ]
        q_acts_text = "activities: " + (", ".join(sorted(prefs.get("act_wants", []))) or "any")
        emb_q = MODEL.encode([q_acts_text], normalize_embeddings=True)
        emb_d = MODEL.encode(docs,           normalize_embeddings=True)
        sims2 = (emb_q @ emb_d.T).flatten()
        boosts = np.where(
            sims2 > SEMANTIC_BOOST_THR,
            (sims2 - SEMANTIC_BOOST_THR) * SEMANTIC_BOOST_EPS,
            0.0
        ).astype("float32")
        scores = cands["score"].to_numpy(copy=True)
        scores[topN_idx] += boosts
        cands["score"] = scores

    cands["explanation"] = [
        f"sim={cands['__sim'].iloc[i]:.2f} | budget={s_budget[i]:.2f} | type={s_type[i]:.2f} | acts={s_act[i]:.2f} | season={s_season[i]:.2f}"
        for i in range(len(cands))
    ]
    cands = cands.sort_values("score", ascending=False)
    if _has("City") and _has("Country"):
        cands = cands.drop_duplicates(subset=[_col("City"), _col("Country")], keep="first")

    # Out cols existentes
    base_out = ["City","Country","Continent","Budget","Season","Tags","Activities","Style"]
    out_cols_real = [c for c in [_col(x) for x in base_out] if c] + ["explanation","score"]
    out_cols_real = list(dict.fromkeys(out_cols_real))
    result = cands[out_cols_real].head(top_k).reset_index(drop=True)

    pf_trace["sizes"]["final_candidates"] = len(result)
    pf_trace["notes"].append(f"used_local_index={used_local_index}")
    if len(result) > 0:
        head_cols_final = [c for c in [_col("City"), _col("Country"), _col("Continent"), _col("Budget"), _col("Season")] if c]
        pf_trace["examples"]["final_head"] = result[head_cols_final].head(5).to_dict(orient="records")
    return result, pf_trace

# ---------- Exports ----------
__all__ = [
    "recommend", "recommend_with_trace", "extract_preferences",
    "jaccard", "season_score", "budget_score",
    "item_types", "item_activities",
    "DF_ROWS", "COLS"
]
