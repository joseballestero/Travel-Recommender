# -*- coding: utf-8 -*-
"""
Evaluación del recomendador (embeddings + reglas)
- Corre un set de prompts (internos o desde CSV)
- Mide efectividad (budget, season, type, acts), pass@k por criterio, diversidad, cobertura
- Soporta ablación de hiperparámetros (alpha, top_k_retrieve, pesos de reglas, boost semántico)
- Mide latencia con warm-up + repeticiones e IC95 por bootstrap
- Exporta resultados (CSV) y un resumen (JSON) para incluir en la memoria

Uso (ejemplos):
  python src/evaluate.py --top-k 10 --runs 5 --warmup 2 --export out/results.csv --show 3
  python src/evaluate.py --runs 5 --warmup 2 --bootstrap 500 --export out/results.csv
  python src/evaluate.py --alpha-list "0.45,0.55,0.65" --topkretrieve-list "150,200" --runs 3 --export out/ablation.csv
  python src/evaluate.py --eval-file data/eval_prompts.csv --top-k 10 --export out/from_csv.csv --show 5
  # override de pesos de reglas (si el módulo recommender los expone):
  python src/evaluate.py --wb 0.35 --wt 0.30 --wa 0.25 --ws 0.10 --export out/weights.csv
"""
import argparse, time, csv, sys, json, os, statistics, random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Import robusto del recomendador ----------
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

try:
    import recommender as R
    from recommender import (
        recommend,               # función principal
        extract_preferences,     # parsea season/budget/type/acts desde el texto
        jaccard, season_score, budget_score,
        item_types, item_activities,
        DF_ROWS                  # dataframe con filas originales
    )
except Exception as e:
    print("ERROR: no se pudo importar 'recommender'. ¿Rutas correctas? ¿Build del índice hecho?")
    print(e)
    sys.exit(1)

# ---------- Prompts por defecto (puedes editar/añadir) ----------
DEFAULT_PROMPTS = [
    "Summer coastal trip, beaches and seafood, relaxing vibe, budget under 120 USD.",
    "Autumn city break focused on museums and architecture, cozy cafés, around 100 USD.",
    "Spring nature escape with hiking and lakes, peaceful vibe, under 90 USD.",
    "Winter city trip for food and nightlife, budget 120 USD.",
    "Family-friendly beach destination, safe, warm weather, budget around 110 USD.",
    "Cultural destination with history, old town, cathedrals, museums, ~100 USD.",
    "Surf-friendly coastal town, relaxed style, budget under 130 USD, summer.",
    "Wine regions with small towns and gastronomy, scenic drives, autumn, ~120 USD.",
    "Island destination with snorkeling/diving, laid-back vibe, summer, budget 140 USD.",
    "Modern city with art galleries, shopping and good restaurants, budget 150 USD.",
    "European coastal city with seafood and beaches, summer, budget under 120 USD.",
    "Nature & hiking near mountains and lakes, spring or summer, budget ~100 USD.",
]

# ---------- Utilidades generales ----------
def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def robust_read_csv(path):
    """Carga CSV tolerante a encoding y líneas problemáticas."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        for enc in ["utf-8-sig", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            except Exception:
                continue
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")

def load_eval_csv(path):
    df = robust_read_csv(path)
    if "query" not in df.columns:
        raise ValueError("El CSV de evaluación debe tener la columna 'query'.")
    return df.to_dict(orient="records")

def parse_list(s):
    if not s or (isinstance(s, float) and np.isnan(s)): return set()
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}

def infer_expectations(row):
    """
    Si el CSV trae columnas expect_* las usa como 'ground truth' de restricción.
    Si no, infiere expectativas usando extract_preferences(query) (consistente con el sistema).
    """
    if any(k in row for k in ["expect_season","expect_type","expect_acts","expect_budget_cap"]):
        season = str(row.get("expect_season","")).strip().lower() or None
        types  = parse_list(row.get("expect_type",""))
        acts   = parse_list(row.get("expect_acts",""))
        bcap   = row.get("expect_budget_cap", None)
        try: bcap = int(bcap)
        except: bcap = None
        return {"season": season, "type_wants": types, "act_wants": acts, "budget_cap": bcap}
    return extract_preferences(row["query"])

# ---------- Métricas auxiliares ----------
def pass_at_k(items, predicate):
    """Proporción de items del top-k que cumplen un predicado binario."""
    if len(items) == 0: return 0.0
    return float(np.mean([1.0 if predicate(r) else 0.0 for _, r in items]))

def continent_diversity_at_k(items):
    """Diversidad por continente en top-k (ratio de continentes únicos / k)."""
    if len(items) == 0: return 0.0
    conts = []
    for _, r in items:
        c = r.get("Continent", "")
        if isinstance(c, str) and c:
            conts.append(c.lower())
    return float(len(set(conts))) / float(len(items))

def acts_coverage_at_k(items, wanted_acts):
    """Cobertura de actividades pedidas: % de acts_wanted presentes al menos una vez en top-k."""
    if not wanted_acts: return 0.0
    found = set()
    for _, r in items:
        idx = DF_ROWS[(DF_ROWS["City"]==r.get("City","")) & (DF_ROWS["Country"]==r.get("Country",""))].index
        base_row = DF_ROWS.loc[idx[0]] if len(idx) else r
        acts_row = item_activities(base_row)
        found |= (wanted_acts & acts_row)
    return float(len(found) / len(wanted_acts))

def failure_buckets_at_k(items, prefs):
    """
    Cuenta fallos por criterio en el top-k:
      - 'budget': presupuesto supera el umbral duro
      - 'season': estación no coincide ni es vecina
      - 'type': Jaccard de tipos = 0 cuando se pedían tipos
      - 'acts': Jaccard de actividades = 0 cuando se pedían acts
    Devuelve dict con conteos y una lista detallada por item.
    """
    buckets = {"budget":0, "season":0, "type":0, "acts":0}
    details = []
    for rank, (_, r) in enumerate(items, start=1):
        b_ok_hard = (budget_score(r.get("Budget",""), prefs["budget_cap"]) >= 1.0) if prefs["budget_cap"] else True
        s_val = season_score(r.get("Season",""), prefs["season"])
        s_ok  = (s_val >= 0.5) if prefs["season"] else True  # exacta o vecina
        # tipos y acts con la fila base
        idx = DF_ROWS[(DF_ROWS["City"]==r.get("City","")) & (DF_ROWS["Country"]==r.get("Country",""))].index
        base_row = DF_ROWS.loc[idx[0]] if len(idx) else r
        types_row = item_types(base_row)
        acts_row  = item_activities(base_row)
        t_j = jaccard(prefs["type_wants"], types_row)
        a_j = jaccard(prefs["act_wants"], acts_row)
        t_ok = (t_j > 0.0) if prefs["type_wants"] else True
        a_ok = (a_j > 0.0) if prefs["act_wants"] else True

        fail = []
        if not b_ok_hard: buckets["budget"] += 1; fail.append("budget")
        if not s_ok:      buckets["season"] += 1; fail.append("season")
        if not t_ok:      buckets["type"]   += 1; fail.append("type")
        if not a_ok:      buckets["acts"]   += 1; fail.append("acts")

        details.append({
            "rank": rank,
            "city": r.get("City",""),
            "country": r.get("Country",""),
            "fails": ",".join(fail) if fail else "",
            "budget_score": float(budget_score(r.get("Budget",""), prefs["budget_cap"])),
            "season_score": float(s_val),
            "type_jaccard": float(t_j),
            "acts_jaccard": float(a_j),
        })
    return buckets, details

# ---------- Medición por query ----------
def measure_query(query_text, top_k=10, runs=3, warmup=1, bootstrap=0):
    """
    Ejecuta recomendación varias veces para latencia estable.
    Devuelve promedios de métricas + distribuciones e IC95 si se piden.
    """
    latencies = []
    results_last = None

    # warm-up
    for _ in range(max(0, warmup)):
        _ = recommend(query_text, top_k=top_k)

    # mediciones
    for _ in range(max(1, runs)):
        t0 = time.time()
        res = recommend(query_text, top_k=top_k)
        latencies.append((time.time() - t0) * 1000.0)
        results_last = res

    res = results_last
    prefs = extract_preferences(query_text)

    if res.empty:
        base = {
            "latency_ms": float(statistics.mean(latencies)),
            "latency_p50": float(np.percentile(latencies,50)),
            "latency_p90": float(np.percentile(latencies,90)),
            "uniq_topk": 0,
            "budget": 0.0, "season": 0.0, "type": 0.0, "acts": 0.0,
            "season_pass_at_k": 0.0, "budget_pass_at_k": 0.0,
            "continent_diversity_at_k": 0.0, "acts_coverage_at_k": 0.0,
            "fail_buckets": {"budget":0,"season":0,"type":0,"acts":0},
            "fail_details": [],
            "results": res, "latency_dist": latencies
        }
        if bootstrap > 0:
            base["ci95"] = {}
        return base

    # métricas existentes sobre todo el top_k
    b_scores, s_scores, t_scores, a_scores = [], [], [], []
    seen = set(); unique = 0
    topk_items = list(res.head(top_k).iterrows())

    for _, r in res.iterrows():
        key = (r.get("City",""), r.get("Country",""))
        if key not in seen:
            unique += 1; seen.add(key)
        b_scores.append(budget_score(r.get("Budget",""), prefs["budget_cap"]))
        s_scores.append(season_score(r.get("Season",""), prefs["season"]))
        idx = DF_ROWS[(DF_ROWS["City"]==r.get("City","")) & (DF_ROWS["Country"]==r.get("Country",""))].index
        base_row = DF_ROWS.loc[idx[0]] if len(idx) else r
        types_row = item_types(base_row)
        acts_row  = item_activities(base_row)
        t_scores.append(jaccard(prefs["type_wants"], types_row))
        a_scores.append(jaccard(prefs["act_wants"], acts_row))

    # métricas @k + diversidad + cobertura + buckets de fallos
    season_atk = pass_at_k(topk_items, lambda r: season_score(r.get("Season",""), prefs["season"]) >= 1.0 if prefs["season"] else True)
    budget_atk = pass_at_k(topk_items, lambda r: budget_score(r.get("Budget",""), prefs["budget_cap"]) >= 1.0 if prefs["budget_cap"] else True)
    cont_div   = continent_diversity_at_k(topk_items)
    acts_cov   = acts_coverage_at_k(topk_items, prefs["act_wants"])
    fb, fb_details = failure_buckets_at_k(topk_items, prefs)

    out = {
        "latency_ms": float(statistics.mean(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p90": float(np.percentile(latencies, 90)),
        "uniq_topk": unique,
        "budget": float(np.mean(b_scores)),
        "season": float(np.mean(s_scores)),
        "type":   float(np.mean(t_scores)),
        "acts":   float(np.mean(a_scores)),
        "season_pass_at_k": float(season_atk),
        "budget_pass_at_k": float(budget_atk),
        "continent_diversity_at_k": float(cont_div),
        "acts_coverage_at_k": float(acts_cov),
        "fail_buckets": fb,
        "fail_details": fb_details,
        "results": res,
        "latency_dist": latencies
    }

    # Bootstrap CI95 sobre métricas principales
    if bootstrap > 0:
        def ci95(vals):
            if len(vals) == 0:
                return (0.0, 0.0)
            bs = []
            for _ in range(bootstrap):
                sample = np.random.choice(vals, size=len(vals), replace=True)
                bs.append(np.mean(sample))
            return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        out["ci95"] = {
            "latency_ms": ci95(latencies),
            "budget":     ci95(b_scores),
            "season":     ci95(s_scores),
            "type":       ci95(t_scores),
            "acts":       ci95(a_scores),
        }
    return out

# ---------- Resumen global ----------
def summarize(rows):
    keys = [
        "latency_ms","latency_p50","latency_p90","uniq_topk",
        "budget","season","type","acts",
        "season_pass_at_k","budget_pass_at_k","continent_diversity_at_k","acts_coverage_at_k"
    ]
    def stat(x):
        return dict(mean=float(np.mean(x)),
                    p50=float(np.percentile(x,50)),
                    p90=float(np.percentile(x,90)))
    agg = {k: [] for k in keys}
    fail_total = {"budget":0,"season":0,"type":0,"acts":0}
    for r in rows:
        for k in keys:
            agg[k].append(r[k])
        # acumula buckets de fallos
        fb = r.get("fail_buckets", {})
        for k in fail_total:
            fail_total[k] += int(fb.get(k, 0))
    summary = {k: stat(v) for k,v in agg.items()}
    summary["fail_buckets_total"] = fail_total
    return summary

# ---------- Overrides de hiperparámetros del recomendador ----------
def maybe_override_recommender(args):
    """
    Permite probar configuraciones sin tocar recommender.py:
      - alpha
      - top_k_retrieve
      - pesos W_BUDGET/W_TYPE/W_ACTIV/W_SEASON
      - boost semántico: topN / eps / thr
    Ignora silenciosamente si alguna variable no existe.
    """
    changed = {}
    try:
        if args.alpha is not None:
            R.ALPHA = float(args.alpha); changed["ALPHA"] = R.ALPHA
    except Exception: pass
    try:
        if args.topk_retrieve is not None:
            R.TOP_K_RETRIEVE = int(args.topk_retrieve); changed["TOP_K_RETRIEVE"] = R.TOP_K_RETRIEVE
    except Exception: pass
    # Pesos de reglas
    def set_if_not_none(name, val):
        if val is None: return
        try:
            setattr(R, name, float(val))
            changed[name] = getattr(R, name)
        except Exception:
            pass
    set_if_not_none("W_BUDGET", args.wb)
    set_if_not_none("W_TYPE",   args.wt)
    set_if_not_none("W_ACTIV",  args.wa)
    set_if_not_none("W_SEASON", args.ws)
    # Boost semántico
    try:
        if args.boost_topn is not None:
            R.SEMANTIC_BOOST_TOPN = int(args.boost_topn); changed["SEMANTIC_BOOST_TOPN"] = R.SEMANTIC_BOOST_TOPN
    except Exception: pass
    set_if_not_none("SEMANTIC_BOOST_EPS", args.boost_eps)
    set_if_not_none("SEMANTIC_BOOST_THR", args.boost_thr)
    return changed

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=str, default="", help="CSV con 'query' (y opcionalmente expect_*).")
    ap.add_argument("--top-k", type=int, default=10, help="k a visualizar/medir del ranking final")
    ap.add_argument("--export", type=str, default="", help="Ruta CSV para exportar métricas por query.")
    ap.add_argument("--show", type=int, default=0, help="Muestra los top-N resultados por cada query en consola.")
    # Repeticiones & CI
    ap.add_argument("--runs", type=int, default=3, help="Repeticiones por query (latencia estable).")
    ap.add_argument("--warmup", type=int, default=1, help="Ejecuciones de calentamiento por query.")
    ap.add_argument("--bootstrap", type=int, default=0, help="Iteraciones bootstrap para IC95 (0=off).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad de resúmenes/IC.")
    # Ablación: listas de alpha y top_k_retrieve
    ap.add_argument("--alpha-list", type=str, default="", help="Barrido de ALPHA, p.ej. '0.45,0.55,0.65'.")
    ap.add_argument("--topkretrieve-list", type=str, default="", help="Barrido de TOP_K_RETRIEVE, p.ej. '150,200'.")
    # Overrides directos de una sola config
    ap.add_argument("--alpha", type=float, default=None, help="ALPHA a usar (override).")
    ap.add_argument("--topk-retrieve", dest="topk_retrieve", type=int, default=None, help="TOP_K_RETRIEVE (override).")
    ap.add_argument("--wb", type=float, default=None, help="Peso budget (override).")
    ap.add_argument("--wt", type=float, default=None, help="Peso type (override).")
    ap.add_argument("--wa", type=float, default=None, help="Peso acts (override).")
    ap.add_argument("--ws", type=float, default=None, help="Peso season (override).")
    ap.add_argument("--boost-topn", type=int, default=None, help="SEMANTIC_BOOST_TOPN (override).")
    ap.add_argument("--boost-eps", type=float, default=None, help="SEMANTIC_BOOST_EPS (override).")
    ap.add_argument("--boost-thr", type=float, default=None, help="SEMANTIC_BOOST_THR (override).")
    args = ap.parse_args()

    set_seeds(args.seed)

    # Carga prompts
    if args.eval_file:
        eval_rows = load_eval_csv(args.eval_file)
        print(f"Loaded {len(eval_rows)} queries from {args.eval_file}")
    else:
        eval_rows = [{"query": q} for q in DEFAULT_PROMPTS]
        print(f"Using {len(eval_rows)} default queries")

    # Ablación de listas; si no se dan, usamos [None] (una sola pasada)
    alpha_vals = [float(x) for x in args.alpha_list.split(",")] if args.alpha_list else [args.alpha]
    topr_vals  = [int(x) for x in args.topkretrieve_list.split(",")] if args.topkretrieve_list else [args.topk_retrieve]

    global_runs = []

    for aval in alpha_vals:
        for kval in topr_vals:
            # Override de hiperparámetros en el módulo del recomendador
            tmp_args = argparse.Namespace(
                alpha=aval, topk_retrieve=kval,
                wb=args.wb, wt=args.wt, wa=args.wa, ws=args.ws,
                boost_topn=args.boost_topn, boost_eps=args.boost_eps, boost_thr=args.boost_thr
            )
            changed = maybe_override_recommender(tmp_args)
            if changed:
                print("Overrides:", changed)

            per_query = []
            out_rows = []
            fail_rows = []

            for i, row in enumerate(eval_rows, 1):
                q = row["query"]
                exp = infer_expectations(row)  # por si la tabla trae expect_*
                print(f"\n[{i}/{len(eval_rows)}] {q}")
                m = measure_query(q, top_k=args.top_k, runs=args.runs, warmup=args.warmup, bootstrap=args.bootstrap)

                per_query.append({
                    "query": q,
                    "latency_ms": m["latency_ms"],
                    "latency_p50": m["latency_p50"],
                    "latency_p90": m["latency_p90"],
                    "uniq_topk": m["uniq_topk"],
                    "budget": m["budget"],
                    "season": m["season"],
                    "type": m["type"],
                    "acts": m["acts"],
                    "season_pass_at_k": m["season_pass_at_k"],
                    "budget_pass_at_k": m["budget_pass_at_k"],
                    "continent_diversity_at_k": m["continent_diversity_at_k"],
                    "acts_coverage_at_k": m["acts_coverage_at_k"],
                })

                # Mostrar top-N por consola si se pide
                if args.show > 0 and not m["results"].empty:
                    for j, (_, r) in enumerate(m["results"].head(args.show).iterrows(), 1):
                        title = f"{r.get('City','')}, {r.get('Country','')}"
                        cont = r.get("Continent", "")
                        title = f"{title} ({cont})" if isinstance(cont, str) and cont else title
                        print(f"  {j:02d}. {title} | Budget={r.get('Budget','n/a')} | Season={r.get('Season','n/a')}")
                        print(f"      Activities={r.get('Activities','')} | Style={r.get('Style','')}")
                        print(f"      {r.get('explanation','')}")
                        out_rows.append({
                            "query": q, "rank": j, "city": r.get("City",""), "country": r.get("Country",""),
                            "continent": r.get("Continent",""), "budget": r.get("Budget",""),
                            "season": r.get("Season",""), "score": float(r.get("score", np.nan)),
                            "explanation": r.get("explanation","")
                        })

                # registrar detalles de fallos (para análisis de errores)
                for d in m["fail_details"]:
                    fail_rows.append({
                        "query": q, "rank": d["rank"], "city": d["city"], "country": d["country"],
                        "fails": d["fails"],
                        "budget_score": d["budget_score"],
                        "season_score": d["season_score"],
                        "type_jaccard": d["type_jaccard"],
                        "acts_jaccard": d["acts_jaccard"],
                    })

            # Resumen
            summ = summarize(per_query)
            print("\n========== SUMMARY ==========")
            for k, s in summ.items():
                if k == "fail_buckets_total":
                    print(f"{k:>24s} | {s}")
                else:
                    print(f"{k:>24s} | mean={s['mean']:.2f} | p50={s['p50']:.2f} | p90={s['p90']:.2f}")

            dfq = pd.DataFrame(per_query)
            print("\nPer-query metrics (head):")
            print(dfq.head(10).to_string(index=False))

            # Export
            if args.export:
                stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                suffix = []
                if aval is not None: suffix.append(f"a{aval}")
                if kval is not None: suffix.append(f"kr{kval}")
                if args.wb is not None: suffix.append(f"wb{args.wb}")
                if args.wt is not None: suffix.append(f"wt{args.wt}")
                if args.wa is not None: suffix.append(f"wa{args.wa}")
                if args.ws is not None: suffix.append(f"ws{args.ws}")
                tag = ("_" + "_".join(suffix)) if suffix else ""
                out_path = Path(args.export).with_name(Path(args.export).stem + f"{tag}.csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                dfq.to_csv(out_path, index=False, encoding="utf-8-sig")

                # Detalles top-N (para tablas en memoria/anexo)
                if out_rows:
                    det_path = out_path.with_name(out_path.stem + "_details.csv")
                    pd.DataFrame(out_rows).to_csv(det_path, index=False, encoding="utf-8-sig")
                # Fallos por item (análisis de errores)
                if fail_rows:
                    fail_path = out_path.with_name(out_path.stem + "_failures.csv")
                    pd.DataFrame(fail_rows).to_csv(fail_path, index=False, encoding="utf-8-sig")

                # JSON con metadatos/config y resumen (ideal para memoria)
                meta = {
                    "timestamp_utc": stamp,
                    "alpha": aval,
                    "top_k_retrieve": kval,
                    "eval_top_k": args.top_k,
                    "runs": args.runs,
                    "warmup": args.warmup,
                    "bootstrap": args.bootstrap,
                    "overrides": {k: getattr(args, k) for k in ["wb","wt","wa","ws","boost_topn","boost_eps","boost_thr"]},
                }
                json_path = out_path.with_name(out_path.stem + "_summary.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"summary": summ, "meta": meta}, f, ensure_ascii=False, indent=2)

                print(f"\nSaved: {out_path}")
                if out_rows:  print(f"       {det_path}")
                if fail_rows: print(f"       {fail_path}")
                print(f"       {json_path}")

            global_runs.append({"alpha": aval, "topk_retrieve": kval, "summary": summ})

if __name__ == "__main__":
    main()
