import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from pettingzoo.classic import rps_v2
from stable_baselines3 import PPO

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def masked_random_action(action_mask: np.ndarray) -> int:
    valid = np.flatnonzero(action_mask)
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))

@dataclass
class BiasedOpponent:
    probs: Tuple[float, float, float]

    def __call__(self, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        a = int(np.random.choice([0, 1, 2], p=self.probs))
        return a if mask[a] == 1 else masked_random_action(mask)

@dataclass
class ModelOpponent:
    model: PPO

    def __call__(self, obs: Dict[str, np.ndarray]) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

class SelfPlayRPS(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, learning_agent: str = "player_0", opponent_policy: Optional[Callable] = None, seed: int = 42):
        super().__init__()
        self.learning_agent = learning_agent
        self.opponent_agent = "player_1" if learning_agent == "player_0" else "player_0"
        self.seed = seed
        self._env = rps_v2.parallel_env()
        self.observation_space = self._env.observation_space(self.learning_agent)
        self.action_space = self._env.action_space(self.learning_agent)
        if opponent_policy is None:
            self._opponent_policy = lambda obs: masked_random_action(obs["action_mask"]) 
        else:
            self._opponent_policy = opponent_policy
        self._obs = None

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is None:
            seed = self.seed
        obs = self._env.reset(seed=seed)
        self._obs = obs
        return obs[self.learning_agent], {}

    def step(self, action: int):
        opp_obs = self._obs[self.opponent_agent]
        opp_action = self._opponent_policy(opp_obs)
        actions = {self.learning_agent: int(action), self.opponent_agent: int(opp_action)}
        obs, rewards, terms, truncs, infos = self._env.step(actions)
        self._obs = obs
        done = bool(terms[self.learning_agent] or truncs[self.learning_agent])
        rew = float(rewards[self.learning_agent])
        info = infos.get(self.learning_agent, {})
        return obs[self.learning_agent], rew, done, False, info

def train_agent(env: gym.Env, timesteps: int, lr: float = 3e-4, seed: int = 42) -> PPO:
    policy = "MultiInputPolicy"
    model = PPO(policy, env, verbose=0, learning_rate=lr, n_steps=64, batch_size=64, seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model

def play_matches(model_A: PPO, model_B: PPO, episodes: int = 500, seed: int = 123):
    env = rps_v2.parallel_env()
    wins = draws = losses = 0
    rng = np.random.RandomState(seed)
    for _ in range(episodes):
        obs = env.reset(seed=int(rng.randint(0, 1_000_000_000)))
        doneA = doneB = False
        while not (doneA or doneB):
            a0, _ = model_A.predict(obs["player_0"], deterministic=True)
            a1, _ = model_B.predict(obs["player_1"], deterministic=True)
            obs, rewards, terms, truncs, _ = env.step({"player_0": int(a0), "player_1": int(a1)})
            doneA = bool(terms["player_0"] or truncs["player_0"])
            doneB = bool(terms["player_1"] or truncs["player_1"])
            if doneA and doneB:
                r = rewards["player_0"]
                if r > 0:
                    wins += 1
                elif r < 0:
                    losses += 1
                else:
                    draws += 1
    n = float(episodes)
    return wins / n, draws / n, losses / n

def evaluate_vs_opponent(model: PPO, opponent: Callable, episodes: int = 200, seed: int = 77) -> float:
    env = SelfPlayRPS(learning_agent="player_0", opponent_policy=opponent, seed=seed)
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(int(a))
            if done and r > 0:
                wins += 1
    return wins / episodes

def main():
    import argparse

    parser = argparse.ArgumentParser(description="rps self-play")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--bursts", type=int, default=5)
    parser.add_argument("--steps-per-burst", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("models", exist_ok=True)

    print("A: train vs biased rock")
    biased_opp = BiasedOpponent(probs=(0.6, 0.2, 0.2))
    env_A = SelfPlayRPS(learning_agent="player_0", opponent_policy=biased_opp, seed=args.seed)
    agent_A = train_agent(env_A, timesteps=args.timesteps, lr=3e-4, seed=args.seed)
    agent_A.save("models/agent_A.zip")
    wrA = evaluate_vs_opponent(agent_A, biased_opp, episodes=300)
    print(f"A: winrate vs biased: {wrA:.3f}")

    print("B: train vs frozen A")
    frozen_A = ModelOpponent(model=agent_A)
    env_B = SelfPlayRPS(learning_agent="player_1", opponent_policy=frozen_A, seed=args.seed + 101)
    agent_B = train_agent(env_B, timesteps=args.timesteps, lr=3e-4, seed=args.seed + 101)
    agent_B.save("models/agent_B.zip")

    print("eval: A vs B")
    winA, draw, lossA = play_matches(agent_A, agent_B, episodes=500)
    print(f"eval: A win/draw/loss: {winA:.3f} / {draw:.3f} / {lossA:.3f}")

    print("curve: B adaptation vs frozen A")
    wr_history = []
    agent_B2 = train_agent(env_B, timesteps=1000, seed=args.seed + 202)
    for k in range(args.bursts):
        wr_B = play_matches(agent_A, agent_B2, episodes=200)[2]
        wr_history.append(wr_B)
        agent_B2.learn(total_timesteps=args.steps_per_burst, progress_bar=False)
        print(f"steps {(k + 1) * args.steps_per_burst}: B winrate vs A = {wr_B:.3f}")
    np.savetxt("models/B_winrate_vs_A.txt", np.array(wr_history))
    print("saved models/B_winrate_vs_A.txt")

if __name__ == "__main__":
    main()