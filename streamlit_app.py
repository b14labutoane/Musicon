import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MOOD_PRESETS, MOOD_LABELS, MOOD_NUMBERS, MOOD_COLORS,
    MOOD_KEYWORDS, FEATURE_COLUMNS, NUM_MOODS,
    SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
)

from recommender import generate_playlist, get_user_songs_from_spotify, apply_feedback, mood_name_to_code, export_playlist_to_spotify
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Musicon", page_icon="\U0001f3b5", layout="wide")

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = None
if "selected_mood" not in st.session_state:
    st.session_state.selected_mood = None
if "playlist" not in st.session_state:
    st.session_state.playlist = None
if "sp_user" not in st.session_state:
    st.session_state.sp_user = None
if "user_songs" not in st.session_state:
    st.session_state.user_songs = None
if "spotify_connected" not in st.session_state:
    st.session_state.spotify_connected = False
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []


def handle_spotify_callback():
    if st.session_state.spotify_connected:
        return

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        params = st.query_params
        code = params.get("code") or params.get("spotify_code", None)
        if not code:
            return

        auth_manager = SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope="user-library-read user-top-read playlist-modify-public",
        )
        token_info = auth_manager.get_access_token(code, as_dict=True)
        if token_info and "access_token" in token_info:
            sp = spotipy.Spotify(auth=token_info["access_token"])
            st.session_state.sp_user = sp
            st.session_state.spotify_connected = True
            with st.spinner("Fetching your Spotify library..."):
                st.session_state.user_songs = get_user_songs_from_spotify(sp)
            n = len(st.session_state.user_songs) if st.session_state.user_songs is not None and not st.session_state.user_songs.empty else 0
            st.session_state.spotify_song_count = n

            if "code" in st.query_params:
                del st.query_params["code"]
            if "spotify_code" in st.query_params:
                del st.query_params["spotify_code"]
    except Exception as e:
        st.error(f"Spotify connection failed: {e}")
        if "code" in st.query_params:
            del st.query_params["code"]
        if "spotify_code" in st.query_params:
            del st.query_params["spotify_code"]


def get_spotify_auth_url():
    from spotipy.oauth2 import SpotifyOAuth
    auth_manager = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="user-library-read user-top-read playlist-modify-public",
    )
    return auth_manager.get_authorize_url()


handle_spotify_callback()


def match_mood_from_caption(text):
    text = text.lower()
    for mood, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return mood
    return "happy"


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1028 35%, #0d1117 100%);
    }
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse at 15% 20%, rgba(91, 94, 166, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 85% 80%, rgba(255, 107, 107, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(107, 203, 119, 0.03) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }
    .stApp > * { position: relative; z-index: 1; }

    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 50%, #6BCB77 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        letter-spacing: -0.03em !important;
        font-size: 3.2rem !important;
    }
    .stCaption {
        font-family: 'JetBrains Mono', monospace !important;
        color: rgba(255,255,255,0.4) !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.02em;
    }
    h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: rgba(255,255,255,0.92) !important;
    }
    h2 { font-weight: 700 !important; }
    h3 { font-weight: 500 !important; font-size: 1.05rem !important; }

    .stButton > button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: rgba(255,255,255,0.85) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(8px) !important;
        padding: 0.55rem 0.9rem !important;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.10) !important;
        border-color: rgba(255,255,255,0.22) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        border-radius: 12px !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.3) !important;
    }
    .stTextInput > div > label {
        color: rgba(255,255,255,0.85) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
    }
    .song-row {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0;
        transition: all 0.2s ease;
    }
    .song-row:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(255,255,255,0.12);
    }
    .mood-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: white !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.4);
    }
    .generate-btn > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 100%) !important;
        color: #0a0a0f !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 14px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 24px rgba(255, 107, 107, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .generate-btn > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.35) !important;
    }
    .streamlit-expanderHeader {
        font-family: 'Outfit', sans-serif !important;
        color: rgba(255,255,255,0.8) !important;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
    }
    .stMarkdown { color: rgba(255,255,255,0.75); }
    .stDivider {
        background: rgba(255,255,255,0.06) !important;
    }
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }
    .stAlert {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    .stPlotlyChart {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }
    .dataframe {
        background: rgba(255,255,255,0.02) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Musicon")
st.caption("AI-powered mood-based playlist generator with 8 moods & reinforcement learning")

with st.expander("Spotify Integration", expanded=not st.session_state.spotify_connected):
    if st.session_state.spotify_connected:
        n = st.session_state.get("spotify_song_count", 0)
        st.success(f"Connected to Spotify! {n} songs matched from your library.")
        if st.button("Disconnect Spotify"):
            st.session_state.sp_user = None
            st.session_state.user_songs = None
            st.session_state.spotify_connected = False
            st.session_state.spotify_song_count = 0
            st.rerun()
    else:
        st.markdown(
            "Connect your Spotify account to personalize playlists based on your listening history."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                auth_url = get_spotify_auth_url()
                st.markdown(
                    f'<a href="{auth_url}" style="'
                    f'display:inline-block; padding:10px 24px; '
                    f'background:#1DB954; color:white; border-radius:24px; '
                    f'text-decoration:none; font-weight:600; font-size:0.9rem;">'
                    f'Connect Spotify</a>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.warning("Spotify credentials not configured.")
        with col_b:
            manual_code = st.text_input(
                "Or paste the code from the callback URL:",
                placeholder="Paste code here...",
                key="manual_spotify_code",
            )
            if manual_code and st.button("Submit Code", key="submit_manual_code"):
                import spotipy
                from spotipy.oauth2 import SpotifyOAuth
                try:
                    clean_code = manual_code.strip()
                    if "?code=" in clean_code:
                        from urllib.parse import urlparse, parse_qs
                        parsed = parse_qs(urlparse(clean_code).query)
                        clean_code = parsed.get("code", [clean_code])[0]
                    auth_manager = SpotifyOAuth(
                        client_id=SPOTIPY_CLIENT_ID,
                        client_secret=SPOTIPY_CLIENT_SECRET,
                        redirect_uri=SPOTIPY_REDIRECT_URI,
                        scope="user-library-read user-top-read playlist-modify-public",
                    )
                    token_info = auth_manager.get_access_token(clean_code, as_dict=True)
                    if token_info and "access_token" in token_info:
                        sp = spotipy.Spotify(auth=token_info["access_token"])
                        st.session_state.sp_user = sp
                        st.session_state.spotify_connected = True
                        with st.spinner("Fetching your Spotify library..."):
                            st.session_state.user_songs = get_user_songs_from_spotify(sp)
                        n_songs = len(st.session_state.user_songs) if st.session_state.user_songs is not None and not st.session_state.user_songs.empty else 0
                        st.session_state.spotify_song_count = n_songs
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

st.markdown("---")
st.header("Choose Your Mood")

preset_names = list(MOOD_PRESETS.keys())
mood_groups = {}
for name in preset_names:
    mood = MOOD_PRESETS[name]["mood"]
    mood_groups.setdefault(mood, []).append(name)

mood_order = [MOOD_LABELS[i] for i in range(NUM_MOODS)]

outer_cols = st.columns(4)
for col_idx, outer_col in enumerate(outer_cols):
    mood_indices = [col_idx, col_idx + 4]
    with outer_col:
        for mi in mood_indices:
            if mi < len(mood_order):
                mood = mood_order[mi]
                presets = mood_groups.get(mood, [])
                for preset in presets:
                    if st.button(preset, key=f"preset_{preset}", use_container_width=True):
                        st.session_state.selected_preset = preset
                        st.session_state.selected_mood = MOOD_PRESETS[preset]["mood"]
                        st.session_state.playlist = None
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

st.markdown("---")

col_text, col_gen = st.columns([3, 1])
with col_text:
    mood_text = st.text_input(
        "Or describe your mood:",
        placeholder="e.g., feeling energetic and want to hit the gym...",
        key="mood_text_input",
    )
with col_gen:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    generate = st.button(
        "Generate Playlist",
        key="generate_btn",
        use_container_width=True,
    )

if generate:
    if st.session_state.selected_preset:
        with st.spinner("Curating your playlist..."):
            st.session_state.playlist = generate_playlist(st.session_state.selected_preset, st.session_state.user_songs)
    elif mood_text:
        detected_mood = match_mood_from_caption(mood_text)
        matching_presets = [p for p, v in MOOD_PRESETS.items() if v["mood"] == detected_mood]
        st.session_state.selected_preset = matching_presets[0] if matching_presets else preset_names[0]
        st.session_state.selected_mood = detected_mood
        with st.spinner("Curating your playlist..."):
            st.session_state.playlist = generate_playlist(st.session_state.selected_preset, st.session_state.user_songs)
    else:
        st.warning("Select a mood preset or describe how you're feeling!")

if st.session_state.selected_preset and not generate:
    if st.session_state.playlist is None:
        with st.spinner("Curating your playlist..."):
            st.session_state.playlist = generate_playlist(st.session_state.selected_preset, st.session_state.user_songs)

playlist = st.session_state.playlist

if playlist:
    mood_name = st.session_state.selected_mood or "happy"
    mood_color = MOOD_COLORS.get(mood_name, "#FFFFFF")
    preset_label = st.session_state.selected_preset or "Custom"

    st.markdown(
        f'<h3 style="color:{mood_color}; margin-bottom:4px;">'
        f'Your Playlist — {preset_label}</h3>',
        unsafe_allow_html=True,
    )
    st.caption(f"{len(playlist)} tracks selected by Musicon")

    for i, song in enumerate(playlist):
        track = song.get("track_name", "Unknown")
        artist = song.get("artist", "Unknown")
        song_mood = song.get("mood", mood_name)
        similarity = song.get("similarity", 0)
        color = MOOD_COLORS.get(song_mood, "#888888")

        sim_pct = f"{similarity * 100:.0f}%" if isinstance(similarity, float) and similarity <= 1 else f"{similarity}"
        genre = song.get("genre", "")
        year = song.get("year", "")
        meta_parts = [p for p in [genre, str(year)] if p]
        meta_str = f" &middot; {' &middot; '.join(meta_parts)}" if meta_parts else ""

        st.markdown(
            f"""
            <div class="song-row">
                <span style="color:rgba(255,255,255,0.3); font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:600; margin-right:12px;">{i + 1:02d}</span>
                <strong style="color:rgba(255,255,255,0.95); font-family:'Outfit',sans-serif; font-size:1rem;">{track}</strong>
                <span style="color:rgba(255,255,255,0.35); margin:0 6px;">—</span>
                <em style="color:rgba(255,255,255,0.55); font-family:'Outfit',sans-serif; font-size:0.88rem;">{artist}</em>
                <span style="color:rgba(255,255,255,0.2); font-size:0.78rem;">{meta_str}</span>
                <span style="float:right; display:flex; align-items:center; gap:8px;">
                    <span style="color:rgba(255,255,255,0.3); font-family:'JetBrains Mono',monospace; font-size:0.72rem;">{sim_pct} match</span>
                    <span class="mood-badge" style="background:{color};">{song_mood}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Rate this playlist to improve future recommendations**")

    feedback_items = []
    cols_per_row = 5
    for row_start in range(0, len(playlist), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(playlist):
                break
            song = playlist[idx]
            short_name = song.get("track_name", "?")[:20]
            song_mood = song.get("mood", mood_name)
            mood_code = mood_name_to_code(song_mood)

            with col:
                rating = st.selectbox(
                    short_name,
                    options=["--", "Like", "Dislike"],
                    key=f"fb_{idx}",
                )
                if rating != "--":
                    feedback_items.append({
                        "position": idx,
                        "mood_code": mood_code,
                        "liked": rating == "Like",
                    })

    if feedback_items:
        fb_col1, fb_col2 = st.columns([1, 3])
        with fb_col1:
            if st.button("Send Feedback", key="submit_feedback", type="primary"):
                avg_delta = apply_feedback(
                    st.session_state.selected_preset or "",
                    feedback_items,
                )
                st.session_state.feedback_log.append(len(feedback_items))
                st.success(
                    f"Feedback applied! {len(feedback_items)} ratings, "
                    f"avg Q-update: {avg_delta:.4f}. "
                    f"Total feedback rounds: {len(st.session_state.feedback_log)}"
                )
        with fb_col2:
            st.info(
                "Your ratings update the Q-Table in real time. "
                "Generate a new playlist to see the difference!"
            )

    if st.session_state.spotify_connected and playlist:
        st.markdown("---")
        export_col1, export_col2 = st.columns([1, 3])
        with export_col1:
            if st.button("Export to Spotify", key="export_spotify_btn", type="primary"):
                with st.spinner("Creating playlist on Spotify..."):
                    try:
                        url, found, not_found = export_playlist_to_spotify(
                            st.session_state.sp_user,
                            playlist,
                            st.session_state.selected_preset or "Musicon",
                        )
                        msg = f"Playlist exported! {found}/{len(playlist)} tracks added."
                        if not_found:
                            msg += f" ({len(not_found)} not found on Spotify)"
                        st.success(msg)
                        if url:
                            st.markdown(f"[Open in Spotify]({url})")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        with export_col2:
            st.info("Create a public playlist on your Spotify account with the generated tracks.")

    with st.expander("Audio Feature Analysis"):
        feature_data = {feat: [] for feat in FEATURE_COLUMNS}
        for song in playlist:
            for feat in FEATURE_COLUMNS:
                val = song.get(feat, 0)
                if val is not None:
                    feature_data[feat].append(float(val))

        avg_values = []
        for feat in FEATURE_COLUMNS:
            vals = feature_data[feat]
            avg_values.append(np.mean(vals) if vals else 0)

        display_names = [f.capitalize() for f in FEATURE_COLUMNS]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=display_names,
            fill="toself",
            fillcolor=mood_color,
            line=dict(color=mood_color, width=2.5),
            opacity=0.7,
            name="Avg Features",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.08)",
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color="rgba(255,255,255,0.7)", family="Outfit"),
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.08)",
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=420,
            margin=dict(t=30, b=30, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        stats_data = {dn: [f"{v:.3f}"] for dn, v in zip(display_names, avg_values)}
        st.markdown("**Average Feature Values**")
        cols = st.columns(len(display_names))
        for ci, (dn, val) in enumerate(zip(display_names, avg_values)):
            with cols[ci]:
                st.markdown(
                    f"<div style='text-align:center; padding:8px 4px; background:rgba(255,255,255,0.03); "
                    f"border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>"
                    f"<div style='font-size:0.65rem; color:rgba(255,255,255,0.4); font-family:JetBrains Mono,monospace; "
                    f"text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;'>{dn}</div>"
                    f"<div style='font-size:1.1rem; font-weight:700; color:{mood_color}; font-family:Outfit,sans-serif;'>"
                    f"{val:.2f}</div></div>",
                    unsafe_allow_html=True,
                )

with st.expander("How it works"):
    st.markdown(
        """
**Musicon** uses a **4-stage AI pipeline** to generate mood-aware playlists from a curated dataset of 10,000+ songs.

**1. Mood Detection**
24 curated presets or free-text keyword matching maps your vibe to 1 of 8 moods.

**2. Song Matching**
Cosine similarity on 8 normalized audio features finds the closest candidates.

**3. Mood Classification**
Decision Tree classifier (99.4% accuracy) assigns mood labels to every track.

**4. Smart Selection**
Q-Learning agent navigates a 192-state MDP to optimize playlist flow and transitions.

---
*Optional: Connect your Spotify account to blend your listening profile into the recommendations.*
""",
    )
