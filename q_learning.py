import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_PROCESSED, FEATURE_COLUMNS, MOOD_LABELS, NUM_MOODS, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, PLAYLIST_LENGTH, CANDIDATE_POOL_SIZE, NUM_POSITION_BUCKETS, NUM_STATES, ALPHA, GAMMA, EPSILON_START, EPSILON_DECAY, EPSILON_MIN, NUM_EPISODES

ADJACENT_MOODS = {
    0: [3, 2],
    1: [7, 0],
    2: [3, 0],
    3: [0, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 7],
    7: [6, 1],
}


def encode_state(current_mood, target_mood, position):
    position_bucket = 0 if position <= 3 else (1 if position <= 7 else 2)
    return current_mood * (NUM_MOODS * NUM_POSITION_BUCKETS) + target_mood * NUM_POSITION_BUCKETS + position_bucket


def compute_reward(song_mood, target_mood):
    if song_mood == target_mood:
        return 1.0
    if song_mood in ADJACENT_MOODS.get(target_mood, []):
        return 0.5
    return 0.0


def get_candidate_pool(train_df, target_mood, rng):
    matching = train_df[train_df["mood_code"] == target_mood]
    if len(matching) >= CANDIDATE_POOL_SIZE:
        pool = matching.sample(n=CANDIDATE_POOL_SIZE, random_state=rng)
    else:
        pool = matching.copy()
        remaining = train_df[train_df["mood_code"] != target_mood]
        needed = CANDIDATE_POOL_SIZE - len(pool)
        if len(remaining) >= needed:
            pool = pd.concat([pool, remaining.sample(n=needed, random_state=rng)])
        else:
            pool = pd.concat([pool, remaining])
    return pool.reset_index(drop=True)


def train():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_path = os.path.join(DATA_PROCESSED, "train.csv")
    train_df = pd.read_csv(train_path)

    Q = np.zeros((NUM_STATES, CANDIDATE_POOL_SIZE))
    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(NUM_EPISODES):
        target_mood = random.randint(0, NUM_MOODS - 1)
        position = 0
        current_mood = target_mood
        total_reward = 0.0

        while position < PLAYLIST_LENGTH:
            state = encode_state(current_mood, target_mood, position)

            pool = get_candidate_pool(train_df, target_mood, rng)

            if rng.random() < epsilon:
                action = rng.randint(0, CANDIDATE_POOL_SIZE)
            else:
                action = int(np.argmax(Q[state]))

            song = pool.iloc[action]
            song_mood = int(song["mood_code"])

            reward = compute_reward(song_mood, target_mood)
            total_reward += reward

            next_position = position + 1
            next_state = encode_state(song_mood, target_mood, next_position)

            best_next = np.max(Q[next_state])
            Q[state, action] += ALPHA * (reward + GAMMA * best_next - Q[state, action])

            current_mood = song_mood
            position = next_position

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

    np.save(os.path.join(MODELS_DIR, "q_table.npy"), Q)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), moving_avg, linewidth=2, label=f"{window}-episode avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning Reward Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reward_curve.png"))
    plt.close(fig)

    avg_last = np.mean(episode_rewards[-window:])
    print(f"Episodes completed: {NUM_EPISODES}")
    print(f"Final epsilon: {epsilon:.4f}")
    print(f"Average reward (last {window} episodes): {avg_last:.2f}")

    return Q, episode_rewards


def main():
    train()


if __name__ == "__main__":
    main()
