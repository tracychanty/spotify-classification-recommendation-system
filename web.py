import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Vibe",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# ── Load models & data ────────────────────────────────────────
@st.cache_resource
def load_models():
    rf         = joblib.load("vibe_classifier.pkl")
    vibe_names = joblib.load("vibe_names.pkl")
    rec_scaler = joblib.load("rec_scaler.pkl")
    rec_matrix = np.load("rec_matrix.npy")
    catalogue  = pd.read_csv("rec_catalogue.csv")
    return rf, vibe_names, rec_scaler, rec_matrix, catalogue

rf, vibe_names, rec_scaler, rec_matrix, catalogue = load_models()

# Feature lists
CLF_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "speechiness", "liveness", "instrumentalness",
    "loudness", "explicit", "duration_min",
]
REC_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "speechiness", "liveness", "instrumentalness",
    "loudness", "duration_min",
]
VIBE_COLORS = {
    "Energetic & Danceable":  "#1DB954",
    "Acoustic & Mellow":      "#3498DB",
    "Instrumental":           "#F39C12",
    "Acoustic & Instrumental":"#9B59B6",
}


# ── Helper functions ──────────────────────────────────────────
def predict_vibe(features: dict):
    df  = pd.DataFrame([features])[CLF_FEATURES]
    pred = rf.predict(df)[0]
    prob = rf.predict_proba(df)[0]
    return pred, pd.Series(prob, index=rf.classes_).sort_values(ascending=False)


def recommend(input_vector, top_n=10, vibe_filter=None, genre_filter=None,
              exclude_idx=None, diverse=False):
    input_vector = np.asarray(input_vector).reshape(1, -1)
    sims         = cosine_similarity(input_vector, rec_matrix).flatten()
    candidate    = catalogue.copy().reset_index(drop=True)
    candidate["similarity"] = sims

    if exclude_idx is not None:
        if isinstance(exclude_idx, int):
            exclude_idx = [exclude_idx]
        candidate = candidate.drop(index=exclude_idx, errors="ignore")

    if vibe_filter:
        if isinstance(vibe_filter, list):
            candidate = candidate[candidate["vibe"].isin(vibe_filter)]
        else:
            candidate = candidate[candidate["vibe"] == vibe_filter]

    if genre_filter:
        if isinstance(genre_filter, list):
            candidate = candidate[candidate["track_genre"].isin(genre_filter)]
        else:
            candidate = candidate[candidate["track_genre"] == genre_filter]

    if candidate.empty:
        return pd.DataFrame()

    if diverse:
        candidate = (
            candidate.sort_values("similarity", ascending=False)
            .groupby("vibe", group_keys=False)
            .head(max(1, top_n // candidate["vibe"].nunique()))
        )

    return (
        candidate.sort_values("similarity", ascending=False)
        .head(top_n)
        [["track_name", "artists", "track_genre", "vibe", "popularity", "similarity"]]
        .reset_index(drop=True)
    )


def recommend_by_song(song_name, top_n=10, same_vibe=False,
                      genre_filter=None, diverse=False):
    matches = catalogue[catalogue["track_name"].str.lower()
                        .str.contains(song_name.lower(), na=False)]
    if matches.empty:
        return None, None

    query      = matches.sort_values("popularity", ascending=False).iloc[0]
    query_idx  = query.name
    query_vibe = query["vibe"]
    query_vector = rec_scaler.transform(
        pd.DataFrame([query[REC_FEATURES]])
    )
    results = recommend(
        query_vector, top_n=top_n,
        vibe_filter=query_vibe if same_vibe else None,
        genre_filter=genre_filter,
        exclude_idx=query_idx,
        diverse=diverse,
    )
    return query, results


def radar_chart(proba_series):
    vibes  = proba_series.index.tolist()
    values = proba_series.values.tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(vibes), endpoint=False).tolist()
    angles += angles[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=vibes + [vibes[0]],
        fill="toself",
        fillcolor="rgba(29,185,84,0.15)",
        line=dict(color="#1DB954", width=2),
        marker=dict(size=6, color="#1DB954"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#141414",
            radialaxis=dict(visible=True, range=[0, 1],
                            color="#444", gridcolor="#222",
                            tickfont=dict(color="#666", size=9)),
            angularaxis=dict(color="#888", gridcolor="#222",
                             tickfont=dict(color="#aaa", size=10)),
        ),
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        margin=dict(l=60, r=60, t=40, b=40),
        showlegend=False,
        height=400,
    )
    return fig


def prob_bar_chart(proba_series):
    fig = go.Figure()
    colors = [VIBE_COLORS.get(v, "#555") for v in proba_series.index]
    fig.add_trace(go.Bar(
        x=proba_series.values,
        y=proba_series.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in proba_series.values],
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(color="#ffffff", size=13, family="DM Sans"),
        outsidetextfont=dict(color="#ffffff", size=13, family="DM Sans"),
        cliponaxis=False,
        constraintext="none",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%", color="#aaaaaa"),
        yaxis=dict(color="#ffffff"),
        uniformtext_minsize=13,
        uniformtext_mode="show",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#141414",
        margin=dict(l=10, r=60, t=10, b=10),
        height=280,
    )
    return fig


def render_rec_results(results: pd.DataFrame):
    if results.empty:
        st.warning("No matching songs found. Try adjusting your filters.")
        return
    for i, row in results.iterrows():
        vibe_color = VIBE_COLORS.get(row["vibe"], "#555")
        st.markdown(f"""
        <div class="result-card">
            <div class="result-rank">#{i+1}</div>
            <div class="result-info">
                <div class="result-name">{row['track_name']}</div>
                <div class="result-artist">{row['artists']} · {row['track_genre']}</div>
            </div>
            <span class="result-badge" style="background:{vibe_color}22;color:{vibe_color}">
                {row['vibe']}
            </span>
            <div class="result-sim">{row['similarity']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"


# ════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # ── Logo (base64 encoded for reliable local rendering) ────
    def img_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    try:
        logo_b64 = img_to_base64("spotify_logo.png")
        st.markdown(
            f'<div style="text-align:center;padding-top:2.5rem;">'
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'width="90" style="display:inline-block;margin-bottom:1rem;">'
            f'</div>',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.markdown('<div style="text-align:center;padding-top:2.5rem;font-size:3rem;">🎵</div>',
                    unsafe_allow_html=True)

    # ── Title + subtitle ──────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;">'
        '<div style="font-family:Syne,sans-serif;font-size:3.8rem;font-weight:800;'
        'background:linear-gradient(135deg,#1DB954,#1ed760,#17a144);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;line-height:1.1;margin-bottom:0.75rem;">'
        'Spotify Vibe</div>'
        '<div style="font-size:1.15rem;color:#888;font-weight:300;margin-bottom:3.5rem;">'
        'Song Vibe Classifier &amp; Discovery Engine</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Role cards ────────────────────────────────────────────
    # Check for card clicks via query params
    params = st.query_params
    if params.get("go") == "artist":
        st.query_params.clear()
        st.session_state.page = "artist"
        st.rerun()
    if params.get("go") == "listener":
        st.query_params.clear()
        st.session_state.page = "listener"
        st.rerun()

    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("""
            <style>
            .role-card { 
                background:#141414; border:1px solid #222; border-radius:16px;
                overflow:hidden; text-align:center; cursor:pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                text-decoration: none !important;
                color: inherit !important;
                display: block;
            }
            .role-card:link, .role-card:visited, .role-card:hover, .role-card:active {
                text-decoration: none !important;
                color: inherit !important;
            }
            .role-card * {
                text-decoration: none !important;
            }
            .role-card:hover {
                transform: translateY(-6px);
                box-shadow: 0 36px 64px rgba(29,185,84,0.55);
                border-color: #1DB954;
            }
            .role-card:hover .card-btn {
                background: linear-gradient(135deg,#22e060,#55ffaa) !important;
            }
            .card-body { padding: 3rem 2.5rem 2rem; }
            .card-icon { font-size:3rem; margin-bottom:1.2rem; }
            .card-title { font-family:Syne,sans-serif; font-size:1.4rem; font-weight:700; color:#fff !important; margin-bottom:0.6rem; }
            .card-desc { color:#888 !important; font-size:0.9rem; line-height:1.6; }
            .card-btn {
                background: linear-gradient(135deg,#1DB954,#32f07a);
                color:#03120a; font-family:Syne,sans-serif; font-weight:700;
                font-size:1.1rem; padding:1.25rem; width:100%;
                box-sizing:border-box; transition: background 0.2s ease;
            }
            </style>
            <a href="?go=artist" target="_self" class="role-card">
                <div class="card-body">
                    <div class="card-icon">🎤</div>
                    <div class="card-title">I'm an Artist</div>
                    <div class="card-desc">Classify your pre-release track and discover which vibe group it belongs to.</div>
                </div>
                <div class="card-btn">Start as Artist</div>
            </a>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <a href="?go=listener" target="_self" class="role-card">
                <div class="card-body">
                    <div class="card-icon">🎧</div>
                    <div class="card-title">I'm a Listener</div>
                    <div class="card-desc">Discover songs that match your mood, energy, or a track you already love.</div>
                </div>
                <div class="card-btn">Start as Listener</div>
            </a>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# ARTIST PAGE — CLASSIFIER
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "artist":

    # ── Artist page header ────────────────────────────────────
    h_col1, h_col2 = st.columns([10, 1])
    with h_col1:
        st.markdown(
            '<div id="top" style="padding:1.2rem 0 0.5rem 0;">'
            '<div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;'
            'color:#1DB954;line-height:1.1;">🎤 Artist Tool</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.markdown("""
        <a href="?go=home" target="_self" style="
            display:flex; align-items:center; justify-content:center;
            width:5rem; height:5rem; border-radius:999px;
            background:linear-gradient(135deg,#1DB954,#32f07a);
            font-size:2rem; text-decoration:none;
            margin-top:0.5rem; margin-left:auto;"
            onmouseover="this.style.transform='translateY(-3px) scale(1.1)'"
            onmouseout="this.style.transform=''">
            🏠
        </a>
        """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Manual Input", "Upload Audio File"])

    # ── Tab 1: Manual sliders ─────────────────────────────────
    with tab1:
        st.markdown("""
        <style>
        div[data-testid="stSlider"] {
            padding-top: 0.4rem !important;
            padding-bottom: 0.8rem !important;
            margin-bottom: 0 !important;
        }
        div[data-testid="stSlider"] > label,
        div[data-testid="stSlider"] label p,
        div[data-testid="stSlider"] p {
            margin-bottom: 0.2rem !important;
            color: #ffffff !important;
        }
        .classify-btn-wrap {
            display: flex;
            justify-content: center;
            margin-top: 1.5rem;
        }
        .classify-btn-wrap .stButton > button {
            width: auto !important;
            min-width: 220px !important;
            padding: 0.9rem 2.5rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.markdown("**Audio Features**")
            danceability = st.slider("Danceability", 0.0, 1.0, 0.5,
                                     help="0 = not danceable, 1 = highly danceable")
            energy = st.slider("Energy", 0.0, 1.0, 0.5,
                               help="0 = calm/quiet, 1 = loud/intense")
            valence = st.slider("Valence (mood)", 0.0, 1.0, 0.5,
                                help="0 = sad/negative, 1 = happy/positive")
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5,
                                     help="0 = electric/produced, 1 = fully acoustic")
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5,
                                         help="0 = has vocals, 1 = purely instrumental")
 
        with col_r:
            st.markdown("**More Features**")
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5,
                                    help="0 = music only, 1 = mostly spoken words")
            liveness = st.slider("Liveness", 0.0, 1.0, 0.5,
                                 help="0 = studio recording, 1 = live performance")
            tempo = st.slider("Tempo (BPM)", 60.0, 220.0, 140.0,
                              help="Estimated beats per minute of the track")
            loudness = st.slider("Loudness (dB)", -40.0, 0.0, -20.0,
                                 help="Overall loudness in dB. Typical range: -40 to 0")
            duration_min = st.slider("Duration (min)", 1.0, 10.0, 5.5,
                                     help="Track length in minutes")
            explicit = 0 

        _, btn_mid, _ = st.columns([1, 1, 1])
        with btn_mid:
            clicked = st.button("Classify My Track", key="classify_manual")

        if clicked:
            features = dict(
                danceability=danceability, energy=energy, valence=valence,
                tempo=tempo, acousticness=acousticness, speechiness=speechiness,
                liveness=liveness, instrumentalness=instrumentalness,
                loudness=loudness, explicit=0, duration_min=duration_min,
            )
            pred, proba = predict_vibe(features)
            vibe_color  = VIBE_COLORS.get(pred, "#1DB954")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem 0">
                <div style="color:#ffffff;font-size:1rem;letter-spacing:0.1em;
                            text-transform:uppercase;margin-bottom:0.5rem;font-weight:600">
                    Predicted Vibe
                </div>
                <div class="vibe-badge" style="background:linear-gradient(135deg,
                    {vibe_color},{vibe_color}aa);font-size:1.6rem;padding:0.8rem 2.5rem;">
                    {pred}
                </div>
                <div style="color:#ffffff;font-size:1.1rem;margin-top:0.75rem;font-weight:500">
                    Confidence: {proba.iloc[0]:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Probability by Vibe**")
                st.plotly_chart(prob_bar_chart(proba), use_container_width=True)
            with c2:
                st.markdown("**Radar Profile**")
                st.plotly_chart(radar_chart(proba), use_container_width=True)

            st.markdown("""
            <a href="#top"
               style="position:fixed; bottom:2rem; right:2rem;
                      width:3.5rem; height:3.5rem; border-radius:999px;
                      background:linear-gradient(135deg,#1DB954,#32f07a);
                      color:#03120a; font-size:1.4rem;
                      display:flex; align-items:center; justify-content:center;
                      text-decoration:none; z-index:9999;
                      box-shadow:0 8px 24px rgba(29,185,84,0.5);
                      transition: transform 0.2s ease, box-shadow 0.2s ease;"
               onmouseover="this.style.transform='translateY(-3px) scale(1.1)';this.style.boxShadow='0 16px 36px rgba(29,185,84,0.7)'"
               onmouseout="this.style.transform='';this.style.boxShadow='0 8px 24px rgba(29,185,84,0.5)'">
               ↑
            </a>
            """, unsafe_allow_html=True)

    # ── Tab 2: Upload audio file ──────────────────────────────
    with tab2:
        uploaded = st.file_uploader("Upload audio file", type=["mp3", "wav"],
                                    label_visibility="collapsed")


# ════════════════════════════════════════════════════════════
# LISTENER PAGE — RECOMMENDER
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "listener":

    if st.query_params.get("go") == "home":
        st.query_params.clear()
        st.session_state.page = "home"
        st.rerun()

    h_col1, h_col2 = st.columns([10, 1])
    with h_col1:
        st.markdown(
            '<div id="top" style="padding:1.2rem 0 0.5rem 0;">'
            '<div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;'
            'color:#1DB954;line-height:1.1;">🎧 Listener Tool</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.markdown("""
        <a href="?go=home" target="_self" style="
            display:flex; align-items:center; justify-content:center;
            width:5rem; height:5rem; border-radius:999px;
            background:linear-gradient(135deg,#1DB954,#32f07a);
            font-size:2rem; text-decoration:none;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-top:0.5rem; margin-left:auto;"
            onmouseover="this.style.transform='translateY(-3px) scale(1.1)';this.style.boxShadow='0 28px 52px rgba(29,185,84,0.70)'"
            onmouseout="this.style.transform='';this.style.boxShadow='0 18px 38px rgba(29,185,84,0.55)'">
            🏠
        </a>
        """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Preference Sliders", "Song Search"])

    # ── Tab 1: Preference sliders ─────────────────────────────
    with tab1:
        st.markdown("""
        <style>
        div[data-testid="stSlider"] {
            padding-top: 0.4rem !important;
            padding-bottom: 0.8rem !important;
            margin-bottom: 0 !important;
        }
        div[data-testid="stSlider"] > label,
        div[data-testid="stSlider"] label p,
        div[data-testid="stSlider"] p {
            margin-bottom: 0.2rem !important;
            color: #ffffff !important;
        }
        .classify-btn-wrap {
            display: flex;
            justify-content: center;
            margin-top: 1.5rem;
        }
        .classify-btn-wrap .stButton > button {
            width: auto !important;
            min-width: 220px !important;
            padding: 0.9rem 2.5rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.markdown("**What's your mood?**")
            p_energy = st.slider("Energy level", 0.0, 1.0, 0.5,
                                      help="0 = very calm, 1 = high energy")
            p_valence = st.slider("Mood (valence)", 0.0, 1.0, 0.5,
                                      help="0 = sad/dark, 1 = happy/positive")
            p_danceability = st.slider("Danceability", 0.0, 1.0, 0.5,
                                      help="0 = not danceable, 1 = highly danceable")
            p_acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5,
                                      help="0 = electronic/produced, 1 = fully acoustic")
 
        with col_r:
            st.markdown("**More preferences**")
            p_tempo = st.slider("Tempo (BPM)", 60.0, 220.0, 140.0,
                                help="Estimated beats per minute of the track")
            p_instrumental = st.slider("Instrumentalness", 0.0, 1.0, 0.5,
                                       help="0 = with vocals, 1 = no vocals")
            p_speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5,
                                      help="0 = music only, 1 = mostly spoken words")
            p_liveness = st.slider("Liveness", 0.0, 1.0, 0.5,
                                   help="0 = studio recording, 1 = live performance")
            p_loudness = st.slider("Loudness (dB)", -40.0, 0.0, -20.0,
                                   help="Overall loudness in dB. Typical range: -40 to 0")
            p_duration = st.slider("Duration (min)", 1.0, 10.0, 5.5,
                                   help="Preferred track length in minutes")

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            top_n_pref = st.selectbox("Results", [5, 10, 15, 20], index=1, key="top_n_pref")
        with col_f2:
            vibe_opts = sorted(catalogue["vibe"].unique())
            vibe_sel  = st.multiselect("Filter by vibe", vibe_opts, placeholder="All vibes", key="vibe_pref")
        with col_f3:
            genre_opts = sorted(catalogue["track_genre"].unique())
            genre_sel  = st.multiselect("Filter by genre", genre_opts, placeholder="All genres", key="genre_pref")

        _, btn_mid, _ = st.columns([2, 1, 2])
        with btn_mid:
            rec_clicked = st.button("Get Recommendations", key="rec_pref_btn")
        if rec_clicked:
            user_prefs = pd.DataFrame([{
                "danceability": p_danceability, "energy": p_energy,
                "valence": p_valence, "tempo": p_tempo,
                "acousticness": p_acousticness, "speechiness": p_speechiness,
                "liveness": p_liveness, "instrumentalness": p_instrumental,
                "loudness": p_loudness, "duration_min": p_duration,
            }])
            user_vector = rec_scaler.transform(user_prefs)
            results = recommend(
                user_vector,
                top_n=top_n_pref,
                vibe_filter=vibe_sel if vibe_sel else None,
                genre_filter=genre_sel if genre_sel else None,
            )
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(f"**{len(results)} recommendations for you:**")
            render_rec_results(results)

            st.markdown("""
            <a href="#top"
               style="position:fixed; bottom:2rem; right:2rem;
                      width:3.5rem; height:3.5rem; border-radius:999px;
                      background:linear-gradient(135deg,#1DB954,#32f07a);
                      color:#03120a; font-size:1.4rem;
                      display:flex; align-items:center; justify-content:center;
                      text-decoration:none; z-index:9999;
                      box-shadow:0 8px 24px rgba(29,185,84,0.5);
                      transition: transform 0.2s ease, box-shadow 0.2s ease;"
               onmouseover="this.style.transform='translateY(-3px) scale(1.1)';this.style.boxShadow='0 16px 36px rgba(29,185,84,0.7)'"
               onmouseout="this.style.transform='';this.style.boxShadow='0 8px 24px rgba(29,185,84,0.5)'">
               ↑
            </a>
            """, unsafe_allow_html=True)

    # ── Tab 2: Song search ────────────────────────────────────
    with tab2:
        st.markdown("""
        <style>
        div[data-testid="stTextInput"] input {
            background: #141414 !important;
            border: 1px solid #444 !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            font-size: 1.1rem !important;
            padding: 0.8rem 1rem !important;
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #666666 !important;
            opacity: 1 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #1DB954 !important;
            box-shadow: 0 0 0 2px rgba(29,185,84,0.2) !important;
        }
        div[data-testid="stToggle"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] label p,
        div[data-testid="stToggle"] span[data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            opacity: 1 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="color:#aaaaaa; font-size:1rem;">
                Enter a song you love and we'll find similar tracks
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, search_col, _ = st.columns([0.5, 4, 0.5])
        with search_col:
            song_query = st.text_input("Song name", placeholder="🔎  e.g. Blinding Lights",
                                       label_visibility="collapsed")

        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

        col_o1, col_o2, col_o3 = st.columns([1, 1, 1])
        with col_o1:
            search_mode      = st.radio(
                "Search mode",
                options=["Diverse vibes", "Same vibe only"],
                horizontal=True,
                label_visibility="visible",
            )
            same_vibe_toggle = (search_mode == "Same vibe only")
            diverse_toggle   = (search_mode == "Diverse vibes")
        with col_o2:
            genre_opts2 = sorted(catalogue["track_genre"].unique())
            genre_sel2  = st.multiselect(
                "Filter by genre",
                genre_opts2,
                placeholder="All genres",
                key="genre_song"
            )
        with col_o3:
            top_n_song = st.selectbox("Results", [5, 10, 15, 20], index=1, key="top_n_song")

        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            search_clicked = st.button("🔎 Find Similar Songs", key="rec_song_btn")

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        if search_clicked:
            if not song_query.strip():
                st.warning("Please enter a song name.")
            else:
                query_row, results = recommend_by_song(
                    song_query,
                    top_n=top_n_song,
                    same_vibe=same_vibe_toggle,
                    genre_filter=genre_sel2 if genre_sel2 else None,
                    diverse=diverse_toggle,
                )
                if query_row is None:
                    st.error(f"Song '{song_query}' not found. Try a different name.")
                else:
                    st.markdown(f"**{len(results)} similar songs:**")
                    render_rec_results(results)

                    st.markdown("""
                    <a href="#top"
                       style="position:fixed; bottom:2rem; right:2rem;
                              width:3.5rem; height:3.5rem; border-radius:999px;
                              background:linear-gradient(135deg,#1DB954,#32f07a);
                              color:#03120a; font-size:1.4rem;
                              display:flex; align-items:center; justify-content:center;
                              text-decoration:none; z-index:9999;">
                       ↑
                    </a>
                    """, unsafe_allow_html=True)