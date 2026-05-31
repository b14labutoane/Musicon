import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from config import DATASET_PATH, DATA_PROCESSED, FEATURE_COLUMNS, MOOD_LABELS, MOOD_NUMBERS, MODELS_DIR, PLAYLIST_LENGTH, CANDIDATE_POOL_SIZE, MOOD_PRESETS, NUM_MOODS, NUM_POSITION_BUCKETS, ALPHA

df = pd.read_csv(DATASET_PATH)

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

Q = np.load(os.path.join(MODELS_DIR, "q_table.npy"))

scaler = joblib.load(os.path.join(DATA_PROCESSED, "scaler.pkl"))

dataset_features_scaled = scaler.transform(df[FEATURE_COLUMNS])


def find_similar_songs(profile: dict, n: int = 50) -> pd.DataFrame:
    profile_vector = np.array([profile[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
    profile_scaled = scaler.transform(profile_vector)
    sims = cosine_similarity(profile_scaled, dataset_features_scaled).flatten()
    df_copy = df.copy()
    df_copy["similarity"] = sims
    return df_copy.nlargest(n, "similarity").copy()


def build_candidate_pool(preset: dict, target_mood: int) -> pd.DataFrame:
    candidates = find_similar_songs(preset, n=100)
    candidates_scaled = scaler.transform(candidates[FEATURE_COLUMNS])
    candidates["predicted_mood"] = model.predict(candidates_scaled).astype(int)

    matching = candidates[candidates["predicted_mood"] == target_mood]
    if len(matching) < 20:
        all_predictions = model.predict(dataset_features_scaled).astype(int)
        all_sims = cosine_similarity(
            scaler.transform(np.array([preset[col] for col in FEATURE_COLUMNS]).reshape(1, -1)),
            dataset_features_scaled,
        ).flatten()
        supplement_df = df.copy()
        supplement_df["predicted_mood"] = all_predictions
        supplement_df["similarity"] = all_sims
        supplement_df = supplement_df[supplement_df["predicted_mood"] == target_mood]
        supplement_df = supplement_df.nlargest(20, "similarity")
        candidates = pd.concat([candidates, supplement_df]).drop_duplicates().copy()

    return candidates


def encode_state(current_mood: int, target_mood: int, position: int) -> int:
    position_bucket = 0 if position <= 3 else (1 if position <= 7 else 2)
    return int(current_mood * 24 + target_mood * 3 + position_bucket)


def generate_playlist(preset_name: str, user_songs: "pd.DataFrame | None" = None) -> list[dict]:
    if preset_name not in MOOD_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = MOOD_PRESETS[preset_name]
    target_mood = MOOD_NUMBERS[preset["mood"]]

    if user_songs is not None and not user_songs.empty:
        user_profile = {}
        for col in FEATURE_COLUMNS:
            user_profile[col] = float(user_songs[col].mean())
        blended = {}
        for col in FEATURE_COLUMNS:
            blended[col] = 0.6 * preset[col] + 0.4 * user_profile[col]
        candidates = build_candidate_pool(blended, target_mood)
    else:
        candidates = build_candidate_pool(preset, target_mood)

    playlist = []
    current_mood = target_mood
    used_indices = set()

    for pos in range(PLAYLIST_LENGTH):
        state = encode_state(current_mood, target_mood, pos)

        available = candidates[~candidates.index.isin(used_indices)]
        if len(available) < CANDIDATE_POOL_SIZE:
            available = candidates.copy()

        target_matches = available[available["predicted_mood"] == target_mood]
        others = available[available["predicted_mood"] != target_mood]

        half = max(CANDIDATE_POOL_SIZE // 2, 1)
        top_target = target_matches.head(half)
        remaining = CANDIDATE_POOL_SIZE - len(top_target)
        top_others = others.head(remaining)
        top_n = pd.concat([top_target, top_others])

        if len(top_n) == 0:
            top_n = available.head(CANDIDATE_POOL_SIZE)

        best_action = int(np.argmax(Q[state]))
        if best_action >= len(top_n):
            best_action = np.random.randint(len(top_n))

        selected = top_n.iloc[best_action]

        entry = {
            "track_name": selected["track_name"],
            "artist": selected["artist"],
            "genre": selected["genre"],
            "year": selected["year"],
            "mood": MOOD_LABELS.get(int(selected["predicted_mood"]), "unknown"),
            "similarity": round(float(selected["similarity"]), 4),
        }
        for col in FEATURE_COLUMNS:
            entry[col] = float(selected[col])

        playlist.append(entry)
        used_indices.add(selected.name)
        current_mood = int(selected["predicted_mood"])

    return playlist

def get_user_songs_from_spotify(sp_user):
    user_songs = []
    results = sp_user.current_user_saved_tracks(limit=50)
    for item in results.get("items", []):
        track = item["track"]
        user_songs.append({"track_name": track["name"], "artist": track["artists"][0]["name"]})

    results = sp_user.current_user_top_tracks(limit=50, time_range="medium_term")
    for track in results.get("items", []):
        user_songs.append({"track_name": track["name"], "artist": track["artists"][0]["name"]})

    seen = set()
    unique = []
    for s in user_songs:
        key = (s["track_name"].lower(), s["artist"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(s)

    matched = []
    for s in unique:
        mask = (df["track_name"].str.lower() == s["track_name"].lower()) & \
               (df["artist"].str.lower() == s["artist"].lower())
        if mask.any():
            matched.append(df.loc[mask].iloc[0])

    return pd.DataFrame(matched) if matched else pd.DataFrame()


def apply_feedback(preset_name: str, feedback: list[dict]) -> float:
    global Q

    if preset_name not in MOOD_PRESETS:
        return 0.0

    preset = MOOD_PRESETS[preset_name]
    target_mood = MOOD_NUMBERS[preset["mood"]]
    alpha = ALPHA

    total_delta = 0.0
    for item in feedback:
        pos = item["position"]
        current_mood = item["mood_code"]
        liked = item["liked"]

        state = encode_state(current_mood, target_mood, pos)
        best_action = int(np.argmax(Q[state]))

        if liked:
            delta = alpha * (1.0 - Q[state, best_action])
            Q[state, best_action] += delta
        else:
            delta = alpha * Q[state, best_action]
            Q[state, best_action] -= delta

        total_delta += abs(delta)

    np.save(os.path.join(MODELS_DIR, "q_table.npy"), Q)

    return total_delta / len(feedback) if feedback else 0.0


def mood_name_to_code(mood_name: str) -> int:
    for code, name in MOOD_LABELS.items():
        if name == mood_name:
            return code
    return -1


def export_playlist_to_spotify(sp_user, playlist: list[dict], preset_name: str = "Musicon"):
    """Export the generated playlist to the user's Spotify account.

    Creates a new public playlist and searches for each track on Spotify.
    Returns the playlist URL on success, raises on failure.
    """
    user_id = sp_user.current_user()["id"]

    sp_playlist = sp_user.user_playlist_create(
        user=user_id,
        name=f"Musicon — {preset_name}",
        public=True,
        description=f"Generated by Musicon | Mood: {preset_name} | {len(playlist)} tracks",
    )

    track_uris = []
    not_found = []

    for song in playlist:
        track_name = song.get("track_name", "")
        artist = song.get("artist", "")
        query = f"track:{track_name} artist:{artist}"

        try:
            results = sp_user.search(q=query, type="track", limit=5)
            items = results.get("tracks", {}).get("items", [])
            if items:
                track_uris.append(items[0]["uri"])
            else:
                broad_query = f"track:{track_name}"
                results = sp_user.search(q=broad_query, type="track", limit=5)
                items = results.get("tracks", {}).get("items", [])
                if items:
                    track_uris.append(items[0]["uri"])
                else:
                    not_found.append(f"{track_name} — {artist}")
        except Exception:
            not_found.append(f"{track_name} — {artist}")

    if track_uris:
        sp_user.user_playlist_add_tracks(user=user_id, playlist_id=sp_playlist["id"], tracks=track_uris)

    url = sp_playlist.get("external_urls", {}).get("spotify", "")
    return url, len(track_uris), not_found

