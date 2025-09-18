# app/streamlit_app.py
import os, sys, re, urllib.parse
from pathlib import Path
import streamlit as st
import pandas as pd  # (puedes quitarlo si no lo usas en otros sitios)

# ---- Robust import of recommender ----
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SRC_DIR.as_posix())

st.set_page_config(
    page_title="Travel Recommender",
    page_icon="🌍",
    layout="centered",
)

# ---------- Styles ----------
st.markdown("""
<style>
.hero {
  background: radial-gradient(1200px 400px at 50% -50%, #eef4ff 0%, #ffffff 60%);
  border: 1px solid #eef2f7; border-radius: 16px; padding: 18px 20px; margin: 10px 0 14px 0;
}
.hero h1 { margin: 0; font-size: 28px; }
.hero p { margin: 6px 0 0 0; color: #4b5563; }

.card {
  border: 1px solid #eaecef; border-radius: 14px; padding: 14px 16px; margin: 10px 0;
  box-shadow: 0 1px 6px rgba(0,0,0,.05); background: #fff;
}
.card h3 { margin: 0 0 2px 0; }
.badges { margin: 4px 0 8px 0; }
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
  background: #f2f6ff; border: 1px solid #e0e7ff; color: #374151; margin-right: 6px;
}
.small { color: #6b7280; font-size: 12px; }
.actions a {
  text-decoration: none; display: inline-block; margin-right: 8px; margin-top: 8px;
  border: 1px solid #e5e7eb; padding: 6px 10px; border-radius: 10px; font-size: 13px;
  background: #fafafa;
}
.actions a:hover { background: #f1f5f9; }

.kv { margin-top: 6px; }
.kv b { color: #111827; }
</style>
""", unsafe_allow_html=True)

# ---------- External links ----------
def booking_url(city: str, country: str) -> str:
    q = f"{city}, {country}".strip()
    return "https://www.booking.com/searchresults.html?ss=" + urllib.parse.quote_plus(q)

def skyscanner_url(_: str = "", __: str = "") -> str:
    """Open Skyscanner main page (no destination prefilled)."""
    return "https://www.skyscanner.net/"

def prettify_list(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "—"
    items = [t.strip() for t in re.split(r"[;,]", text) if t.strip()]
    if not items:
        return "—"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

# ---------- Load recommender ----------
try:
    from recommender import recommend
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    st.error("The recommender could not be loaded.")
    st.caption("Build the FAISS index first and ensure paths are correct.")
    with st.expander("Error details"):
        st.exception(e)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <h1>🌍 Travel Recommender</h1>
  <p>Skip the endless searching. Your perfect destination is one question away. 
    Just ask, and get your travel match instantly.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Options")
    top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10, step=1)
    st.divider()
    st.subheader("Examples")
    examples = [
        "Summer coastal trip with seafood and beaches, relaxing vibe, under 120 USD in Europe.",
        "Autumn city break with museums and architecture, around 100 USD in North America.",
        "Nature & hiking near mountains and lakes, spring or summer, budget ~100 USD in Asia.",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")

# ---------- Main form ----------
if artifacts_ok:
    with st.form("query_form"):
        user_text = st.text_area(
            "What are you looking for?",
            value="",
            height=110,
            placeholder="Tell me what you like"
        )
        col1, col2 = st.columns([1,1])
        with col1:
            submitted = st.form_submit_button("🔎 Recommend", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🧹 Clear", use_container_width=True)

    if clear:
        st.experimental_set_query_params()
        st.rerun()

    if submitted:
        if not user_text.strip():
            st.warning("Please type your trip idea.")
        else:
            try:
                res = recommend(user_text, top_k=top_k)

                if res.empty:
                    st.info("No destinations matched your request. Try loosening constraints or changing the description.")
                else:
                    st.write("### Results")
                    for _, r in res.iterrows():
                        city = str(r.get("City", "")).strip()
                        country = str(r.get("Country", "")).strip()
                        continent = r.get("Continent", "")
                        title = f"{city}, {country}" + (f" ({continent})" if isinstance(continent, str) and continent else "")

                        budget = r.get("Budget", "n/a")
                        season = r.get("Season", "n/a")
                        acts = prettify_list(r.get("Activities", ""))
                        style = prettify_list(r.get("Style", ""))

                        url_booking = booking_url(city, country)
                        url_skyscanner = skyscanner_url()  # <-- home de Skyscanner

                        st.markdown(f"""
                        <div class="card">
                          <h3>{title}</h3>
                          <div class="badges">
                            <span class="badge">Budget: {budget}</span>
                            <span class="badge">Best season: {season}</span>
                          </div>
                          <div class="kv"><b>Activities:</b> {acts}</div>
                          <div class="kv"><b>Vibe:</b> {style}</div>
                          <div class="actions">
                            <a href="{url_skyscanner}" target="_blank">✈️ Open Skyscanner</a>
                            <a href="{url_booking}" target="_blank">🏨 Find stays (Booking)</a>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.caption("Tip: you can include a continent in your description (e.g., 'in Europe', 'in Asia').")
            except Exception as e:
                st.error("An error occurred while generating recommendations.")
                with st.expander("Error details"):
                    st.exception(e)
else:
    st.info("Artifacts missing. Build the index first:")
    st.code("python src/build_index.py --data data/Destinations.csv --out data")

st.caption("© Travel Recommender")
