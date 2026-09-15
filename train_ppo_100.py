"""Run and record 100 PPO exploration episodes."""

import csv
import json
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_gym_env import NarutoMarchingGymEnv


EPISODE_COUNT = 100
MAX_STEPS_PER_EPISODE = 1000
TOTAL_TIMESTEPS = EPISODE_COUNT * MAX_STEPS_PER_EPISODE
OUTPUT_JSON = Path("ppo_100_episode_results_occupied.json")
OUTPUT_CSV = Path("ppo_100_episode_results_occupied.csv")


class EpisodeLimit(gym.Wrapper):
    """Stop an episode after a fixed number of actions and expose final state."""

    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if terminated or truncated or self.steps >= self.max_steps:
            info = dict(info)
            info["final_score"] = int(self.env.engine.total_reward)
            info["final_food"] = int(self.env.engine.current_food)
            info["final_day"] = int(self.env.engine.current_day)
            info["final_occupied"] = len(self.env.engine.all_visited_hexes)
            info["episode_steps"] = self.steps
            if self.steps >= self.max_steps and not terminated:
                truncated = True
        return observation, reward, terminated, truncated, info


class EpisodeRecorder(BaseCallback):
    def __init__(self, target_episodes):
        super().__init__()
        self.target_episodes = target_episodes
        self.results = []
        self.started = time.perf_counter()

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if not done or "final_score" not in info:
                continue
            result = {
                "episode": len(self.results) + 1,
                "score": int(info["final_score"]),
                "training_reward": float(info.get("episode", {}).get("r", 0.0)),
                "final_day": int(info["final_day"]),
                "final_food": int(info["final_food"]),
                "final_occupied": int(info["final_occupied"]),
                "steps": int(info["episode_steps"]),
                "elapsed_seconds": round(time.perf_counter() - self.started, 6),
            }
            self.results.append(result)
            print(
                f"episode={result['episode']:03d} score={result['score']} "
                f"day={result['final_day']} food={result['final_food']} "
                f"steps={result['steps']}"
            )
            if len(self.results) >= self.target_episodes:
                return False
        return True


def main():
    env = DummyVecEnv([lambda: EpisodeLimit(NarutoMarchingGymEnv(), MAX_STEPS_PER_EPISODE)])
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        seed=42,
    )
    recorder = EpisodeRecorder(EPISODE_COUNT)
    started = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=recorder)
    elapsed = time.perf_counter() - started

    payload = {
        "episodes_requested": EPISODE_COUNT,
        "episodes_completed": len(recorder.results),
        "total_timesteps": int(model.num_timesteps),
        "elapsed_seconds": round(elapsed, 6),
        "results": recorder.results,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=recorder.results[0].keys() if recorder.results else [])
        if recorder.results:
            writer.writeheader()
            writer.writerows(recorder.results)

    print(f"completed={len(recorder.results)}/{EPISODE_COUNT}")
    print(f"timesteps={model.num_timesteps}")
    print(f"elapsed_seconds={elapsed:.6f}")
    if recorder.results:
        scores = [row["score"] for row in recorder.results]
        print(f"score_first={scores[0]}")
        print(f"score_last={scores[-1]}")
        print(f"score_best={max(scores)}")
        print(f"score_mean={sum(scores) / len(scores):.3f}")
    env.close()


if __name__ == "__main__":
    main()
