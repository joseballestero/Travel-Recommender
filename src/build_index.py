#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reconstruye los artefactos del buscador a partir de data/Destinations.csv:
- data/faiss.index
- data/id_map.npy
- data/rows.pkl
- data/index_meta.json

Estructura de carpetas esperada:
  project_root/
    data/
      Destinations.csv
    src/
      build_index.py
      recommender.py

Puedes ejecutar sin argumentos desde project_root o desde src; detecta rutas solo.
"""

import os
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# --------- Paths por defecto basados en la ubicación del script ---------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent                     # ../
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"           # ../data
DEFAULT_CSV = DEFAULT_DATA_DIR / "Destinations.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED_COLS = [
    "City", "Country", "Continent", "Season", "Budget",
    "Tags", "Activities", "Style", "FullText", "is_coastal"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Asegura columnas básicas
    for col in ["Tags","Activities","Style"]:
        if col not in df.columns: df[col] = ""
    # is_coastal: si no existe, infiérela débilmente de Tags/Style
    if "is_coastal" not in df.columns:
        pat = r"\b(coastal|seaside|oceanfront|waterfront|shore|bay|harbor|harbour)\b"
        guess = pd.Series(False, index=df.index)
        if "Tags" in df.columns:
            guess = guess | df["Tags"].fillna("").str.contains(pat, case=False, regex=True, na=False)
        if "Style" in df.columns:
            guess = guess | df["Style"].fillna("").str.contains(pat, case=False, regex=True, na=False)
        df["is_coastal"] = guess.astype(bool)

    # FullText si falta → concat “limpia” por filas
    if "FullText" not in df.columns or df["FullText"].isna().all():
        cols = [c for c in ["City","Country","Continent","Season","Budget","Tags","Activities","Style"] if c in df.columns]
        if cols:
            df["FullText"] = df[cols].fillna("").astype(str).agg(" | ".join, axis=1)
        else:
            df["FullText"] = ""

    # Tipos y espacios
    for c in ["City","Country","Continent","Season","Budget","Tags","Activities","Style","FullText"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Normaliza continente y season (minúsculas)
    if "Continent" in df.columns:
        df["Continent"] = df["Continent"].str.lower()
    if "Season" in df.columns:
        df["Season"] = df["Season"].str.lower()

    return df

def make_corpus_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = "" if c != "is_coastal" else False
    return out[REQUIRED_COLS]

def make_text_for_embedding(row: pd.Series) -> str:
    return " | ".join([
        str(row.get("Tags","")),
        str(row.get("Activities","")),
        str(row.get("Style","")),
        str(row.get("FullText","")),
    ])

def build(csv_path: Path, data_dir: Path, model_name: str, batch_size: int = 512):
    print(f"==> CSV: {csv_path}")
    print(f"==> DATA_DIR destino: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = normalize_cols(df)

    # Validación mínima
    for col in ["City","Country","Continent"]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria '{col}' en el CSV.")

    rows = make_corpus_rows(df)
    print(f"Filas: {len(rows)}")
    print("Continentes únicos:", sorted(rows["Continent"].unique()))

    print(f"==> Cargando modelo: {model_name}")
    model = SentenceTransformer(model_name)

    print("==> Construyendo corpus textual…")
    corpus = rows.apply(make_text_for_embedding, axis=1).tolist()

    print("==> Calculando embeddings…")
    emb = model.encode(corpus, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]
    print(f"Embeddings shape: {emb.shape}")

    print("==> Construyendo índice FAISS…")
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    id_map = np.arange(len(rows), dtype=np.int64)

    print("==> Guardando artefactos…")
    faiss.write_index(index, str(data_dir / "faiss.index"))
    np.save(str(data_dir / "id_map.npy"), id_map)
    rows.to_pickle(data_dir / "rows.pkl")
    meta = {
        "model": model_name,
        "dim": int(dim),
        "ntotal": int(index.ntotal),
        "source_csv": str(csv_path.resolve()),
        "schema": REQUIRED_COLS,
    }
    with open(data_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n==> Sanity checks")
    print(" - Artefactos escritos en:", str(data_dir.resolve()))
    print(" - Columnas en rows.pkl:", list(rows.columns))
    print(" - Continentes:", sorted(rows["Continent"].unique()))
    print(" - Ejemplo:", rows.iloc[0].to_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Ruta a Destinations.csv")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Directorio de salida (data)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo SentenceTransformer")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    build(
        csv_path=Path(args.csv),
        data_dir=Path(args.data_dir),
        model_name=args.model,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
