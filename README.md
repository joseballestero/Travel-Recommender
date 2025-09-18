# 🌍 Travel Recommender

AI-powered destination finder that turns natural-language wishes into actionable suggestions.  
It blends FAISS vector search with structured rules (budget, season, coastal, continent, activities).

---

## Features
- **Natural-language queries**: just type what you want.
- **Hard constraints respected**: continent (strict), coastal include/exclude, budget cap, and season (with neighbor season fallback).
- **Hybrid ranking**: FAISS similarity blended with rules (α=0.55; weights for budget/type/activities/season = 0.32/0.28/0.30/0.10).
- **Spelling-noise robustness**: remains useful with typos.
- **Fast on CPU**: typical mean latency ≈ 140 ms in tests.
- **Actionable links**: open Skyscanner homepage for flights and Booking.com city search for stays.
- **No GPU / minimal ops**: FAISS flat index + Streamlit UI.

---

## Quickstart (local)

> Requires Python 3.10+.

```bash
# 1) Clone
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2) (Optional) Create & activate a virtual env/conda env
# python -m venv .venv && . .venv/Scripts/activate            # Windows (PowerShell)
# python3 -m venv .venv && source .venv/bin/activate           # macOS/Linux
# or: conda create -n tfm-viajes python=3.10 && conda activate tfm-viajes

# 3) Install deps
pip install -r requirements.txt

# 4) Build the FAISS index (uses data/Destinations.csv)
python src/build_index.py --data data/Destinations.csv --out data

# 5) Run the app
streamlit run app/streamlit_app.py
