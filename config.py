import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_PATH = os.path.join(DATA_RAW, "popular_10k_mood_labeled.csv")

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
]

MOOD_LABELS = {
    0: "melancholic",
    1: "dark",
    2: "calm",
    3: "nostalgic",
    4: "romantic",
    5: "happy",
    6: "euphoric",
    7: "energetic",
}

MOOD_NUMBERS = {name: code for code, name in MOOD_LABELS.items()}
NUM_MOODS    = len(MOOD_LABELS)

MOOD_COLORS = {
    "melancholic": "#5B5EA6",
    "dark"       : "#3D3D3D",
    "calm"       : "#4D96FF",
    "nostalgic"  : "#C68B59",
    "romantic"   : "#E8747C",
    "happy"      : "#FFD93D",
    "euphoric"   : "#FF6B6B",
    "energetic"  : "#6BCB77",
}

MOOD_PRESETS = {
    # melancholic
    "Heartbreak": {
        "mood": "melancholic",
        "danceability": 0.30, "energy": 0.30, "loudness": 0.50,
        "speechiness": 0.05, "acousticness": 0.60, "instrumentalness": 0.20,
        "valence": 0.15, "tempo": 0.30,
    },
    "Rainy window": {
        "mood": "melancholic",
        "danceability": 0.25, "energy": 0.25, "loudness": 0.40,
        "speechiness": 0.05, "acousticness": 0.70, "instrumentalness": 0.35,
        "valence": 0.20, "tempo": 0.25,
    },
    "Fading light": {
        "mood": "melancholic",
        "danceability": 0.35, "energy": 0.40, "loudness": 0.55,
        "speechiness": 0.10, "acousticness": 0.45, "instrumentalness": 0.10,
        "valence": 0.18, "tempo": 0.40,
    },
    # dark
    "Brooding": {
        "mood": "dark",
        "danceability": 0.40, "energy": 0.65, "loudness": 0.75,
        "speechiness": 0.15, "acousticness": 0.05, "instrumentalness": 0.00,
        "valence": 0.15, "tempo": 0.70,
    },
    "Late night": {
        "mood": "dark",
        "danceability": 0.35, "energy": 0.60, "loudness": 0.70,
        "speechiness": 0.10, "acousticness": 0.10, "instrumentalness": 0.10,
        "valence": 0.10, "tempo": 0.65,
    },
    "Angry & frustrated": {
        "mood": "dark",
        "danceability": 0.45, "energy": 0.85, "loudness": 0.90,
        "speechiness": 0.25, "acousticness": 0.02, "instrumentalness": 0.00,
        "valence": 0.10, "tempo": 0.85,
    },
    # calm
    "Morning coffee": {
        "mood": "calm",
        "danceability": 0.40, "energy": 0.20, "loudness": 0.40,
        "speechiness": 0.05, "acousticness": 0.65, "instrumentalness": 0.40,
        "valence": 0.65, "tempo": 0.25,
    },
    "Winding down": {
        "mood": "calm",
        "danceability": 0.25, "energy": 0.15, "loudness": 0.35,
        "speechiness": 0.03, "acousticness": 0.75, "instrumentalness": 0.55,
        "valence": 0.55, "tempo": 0.20,
    },
    "Study session": {
        "mood": "calm",
        "danceability": 0.35, "energy": 0.20, "loudness": 0.45,
        "speechiness": 0.05, "acousticness": 0.60, "instrumentalness": 0.50,
        "valence": 0.50, "tempo": 0.30,
    },
    # nostalgic
    "Nostalgia": {
        "mood": "nostalgic",
        "danceability": 0.40, "energy": 0.35, "loudness": 0.45,
        "speechiness": 0.08, "acousticness": 0.65, "instrumentalness": 0.25,
        "valence": 0.40, "tempo": 0.35,
    },
    "Old memories": {
        "mood": "nostalgic",
        "danceability": 0.35, "energy": 0.30, "loudness": 0.40,
        "speechiness": 0.05, "acousticness": 0.70, "instrumentalness": 0.30,
        "valence": 0.35, "tempo": 0.30,
    },
    "Night drive": {
        "mood": "nostalgic",
        "danceability": 0.45, "energy": 0.40, "loudness": 0.55,
        "speechiness": 0.05, "acousticness": 0.45, "instrumentalness": 0.20,
        "valence": 0.38, "tempo": 0.45,
    },
    # romantic
    "Romantic dinner": {
        "mood": "romantic",
        "danceability": 0.35, "energy": 0.25, "loudness": 0.40,
        "speechiness": 0.04, "acousticness": 0.55, "instrumentalness": 0.40,
        "valence": 0.55, "tempo": 0.25,
    },
    "Date night": {
        "mood": "romantic",
        "danceability": 0.40, "energy": 0.35, "loudness": 0.45,
        "speechiness": 0.05, "acousticness": 0.45, "instrumentalness": 0.25,
        "valence": 0.50, "tempo": 0.30,
    },
    "Wine night": {
        "mood": "romantic",
        "danceability": 0.38, "energy": 0.28, "loudness": 0.42,
        "speechiness": 0.04, "acousticness": 0.50, "instrumentalness": 0.35,
        "valence": 0.52, "tempo": 0.28,
    },
    # happy
    "Feeling good": {
        "mood": "happy",
        "danceability": 0.65, "energy": 0.65, "loudness": 0.75,
        "speechiness": 0.10, "acousticness": 0.20, "instrumentalness": 0.00,
        "valence": 0.80, "tempo": 0.55,
    },
    "Road trip": {
        "mood": "happy",
        "danceability": 0.70, "energy": 0.60, "loudness": 0.70,
        "speechiness": 0.08, "acousticness": 0.30, "instrumentalness": 0.05,
        "valence": 0.75, "tempo": 0.50,
    },
    "Beach vibes": {
        "mood": "happy",
        "danceability": 0.65, "energy": 0.55, "loudness": 0.65,
        "speechiness": 0.08, "acousticness": 0.35, "instrumentalness": 0.00,
        "valence": 0.78, "tempo": 0.45,
    },
    # euphoric
    "Dance party": {
        "mood": "euphoric",
        "danceability": 0.85, "energy": 0.85, "loudness": 0.85,
        "speechiness": 0.10, "acousticness": 0.05, "instrumentalness": 0.00,
        "valence": 0.85, "tempo": 0.75,
    },
    "Celebration": {
        "mood": "euphoric",
        "danceability": 0.80, "energy": 0.80, "loudness": 0.82,
        "speechiness": 0.12, "acousticness": 0.05, "instrumentalness": 0.00,
        "valence": 0.88, "tempo": 0.70,
    },
    "Vegas trip": {
        "mood": "euphoric",
        "danceability": 0.80, "energy": 0.88, "loudness": 0.88,
        "speechiness": 0.15, "acousticness": 0.03, "instrumentalness": 0.00,
        "valence": 0.82, "tempo": 0.72,
    },
    # energetic
    "Gym workout": {
        "mood": "energetic",
        "danceability": 0.60, "energy": 0.92, "loudness": 0.90,
        "speechiness": 0.15, "acousticness": 0.03, "instrumentalness": 0.00,
        "valence": 0.40, "tempo": 0.85,
    },
    "Going for a run": {
        "mood": "energetic",
        "danceability": 0.65, "energy": 0.90, "loudness": 0.88,
        "speechiness": 0.12, "acousticness": 0.05, "instrumentalness": 0.00,
        "valence": 0.45, "tempo": 0.80,
    },
    "Rock out": {
        "mood": "energetic",
        "danceability": 0.50, "energy": 0.90, "loudness": 0.90,
        "speechiness": 0.10, "acousticness": 0.03, "instrumentalness": 0.00,
        "valence": 0.50, "tempo": 0.80,
    },
}

MOOD_KEYWORDS = {
    "melancholic": [
        "heartbreak", "lonely", "miss", "missing", "cry", "alone",
        "breakup", "sad", "hurt", "pain", "tears", "lost", "fading",
    ],
    "dark": [
        "angry", "frustrated", "dark", "brooding", "hate", "rage",
        "intense", "heavy", "gritty", "night", "moody",
    ],
    "calm": [
        "study", "rain", "chill", "sleep", "relax", "cozy", "tea",
        "reading", "focus", "zen", "calm", "peace", "quiet", "coffee",
        "morning", "winding down", "meditation",
    ],
    "nostalgic": [
        "nostalgia", "memories", "old", "past", "remember", "reminisce",
        "childhood", "vintage", "throwback", "retro", "drive",
    ],
    "romantic": [
        "romantic", "love", "date", "dinner", "wine", "candle",
        "tender", "sweetheart", "valentine", "couple", "kiss",
    ],
    "happy": [
        "happy", "good", "great", "excited", "beach", "road trip",
        "friends", "celebration", "joy", "fun", "smile", "sunshine",
        "awesome", "amazing", "blessed", "summer",
    ],
    "euphoric": [
        "party", "dance", "club", "vegas", "celebrate", "ecstatic",
        "lit", "turn up", "festival", "rave", "wild",
    ],
    "energetic": [
        "run", "running", "gym", "workout", "pump", "explode",
        "adrenaline", "power", "energy", "hype", "rock", "gaming",
        "beast", "sweat", "lift", "cardio",
    ],
}

RANDOM_SEED = 42
TEST_SIZE = 0.2

PLAYLIST_LENGTH = 10
CANDIDATE_POOL_SIZE = 10
NUM_POSITION_BUCKETS = 3
NUM_STATES = NUM_MOODS*NUM_MOODS*NUM_POSITION_BUCKETS

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
NUM_EPISODES = 2000

SPOTIPY_CLIENT_ID = "cfead4ec16aa42e081e05bdc5942da34" 
SPOTIPY_CLIENT_SECRET = "fe3508c6cfa3427783cb1594afc774b2"
SPOTIPY_REDIRECT_URI = "https://b14labutoane.github.io/musicon-callback/"

