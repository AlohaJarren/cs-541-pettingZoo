import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
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
    v = np.flatnonzero(action_mask)
    if len(v) == 0:
        return 0
    return int(np.random.choice(v))

def get_mask(o) -> np.ndarray:
    if isinstance(o, dict):
        m = o.get("action_mask", None)
        if m is None:
            return np.ones(3, dtype=np.float32)
        return np.array(m, dtype=np.float32).flatten()
    return np.ones(3, dtype=np.float32)

def encode_obs(o) -> np.ndarray:
    if isinstance(o, dict):
        obs = np.array(o.get("observation", []), dtype=np.float32).flatten()
        if obs.size != 3:
            obs = np.zeros(3, dtype=np.float32)
        mask = get_mask(o)
        if mask.size != 3:
            mask = np.ones(3, dtype=np.float32)
        return np.concatenate([obs, mask]).astype(np.float32)
    return np.concatenate([np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)]).astype(np.float32)

def pz_reset(env, seed=None):
    res = env.reset(seed=seed)
    if isinstance(res, tuple):
        return res[0], res[1]
    return res, {}

@dataclass
class BiasedOpponent:
    probs: Tuple[float, float, float]

    def __call__(self, obs: Dict[str, np.ndarray]) -> int:
        mask = get_mask(obs)
        a = int(np.random.choice([0, 1, 2], p=self.probs))
        return a if mask[a] == 1 else masked_random_action(mask)

@dataclass
class ModelOpponent:
    model: PPO
    deterministic: bool = False

    def __call__(self, obs: Dict[str, np.ndarray]) -> int:
        a, _ = self.model.predict(encode_obs(obs), deterministic=self.deterministic)
        return int(a)


class SelfPlayRPS(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, learning_agent: str = "player_0", opponent_policy: Optional[Callable] = None, seed: int = 42):
        super().__init__()
        self.learning_agent = learning_agent
        self.opponent_agent = "player_1" if learning_agent == "player_0" else "player_0"
        self.seed = seed
        self._env = rps_v2.parallel_env()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        if opponent_policy is None:
            self._opponent_policy = lambda o: masked_random_action(get_mask(o))
        else:
            self._opponent_policy = opponent_policy
        self._obs = None

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is None:
            seed = self.seed
        obs, _info = pz_reset(self._env, seed=seed)
        self._obs = obs
        return encode_obs(obs[self.learning_agent]), {}

    def step(self, action: int):
        opp_obs = self._obs[self.opponent_agent]
        opp_action = self._opponent_policy(opp_obs)
        actions = {self.learning_agent: int(action), self.opponent_agent: int(opp_action)}
        obs, rewards, terms, truncs, infos = self._env.step(actions)
        self._obs = obs
        done = bool(terms[self.learning_agent] or truncs[self.learning_agent])
        rew = float(rewards[self.learning_agent])
        info = infos.get(self.learning_agent, {})
        return encode_obs(obs[self.learning_agent]), rew, done, False, info


def make_ppo(env: gym.Env, seed: int, ent: float) -> PPO:
    return PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=64, batch_size=64, seed=seed, ent_coef=ent)


def train_agent(env: gym.Env, timesteps: int, seed: int, ent: float) -> PPO:
    model = make_ppo(env, seed, ent)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model

def play_matches(model_A: PPO, model_B: PPO, episodes: int = 300, seed: int = 123, deterministic: bool = False):
    env = rps_v2.parallel_env()
    wins = draws = losses = 0
    rng = np.random.RandomState(seed)
    for _ in range(episodes):
        obs, _info = pz_reset(env, seed=int(rng.randint(0, 1_000_000_000)))
        doneA = doneB = False
        while not (doneA or doneB):
            a0, _ = model_A.predict(encode_obs(obs["player_0"]), deterministic=deterministic)
            a1, _ = model_B.predict(encode_obs(obs["player_1"]), deterministic=deterministic)
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

def evaluate_vs_opponent(model: PPO, opponent: Callable, episodes: int = 200, seed: int = 77, deterministic: bool = False) -> float:
    env = SelfPlayRPS(learning_agent="player_0", opponent_policy=opponent, seed=seed)
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _, _ = env.step(int(a))
            if done and r > 0:
                wins += 1
    return wins / episodes

def main():
    import argparse

    p = argparse.ArgumentParser(description="rps self-play switches")
    p.add_argument("--timesteps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--entropy", type=float, default=0.03)
    p.add_argument("--switches", type=int, default=20)
    p.add_argument("--switch-steps", type=int, default=4000)
    p.add_argument("--eval-episodes", type=int, default=300)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs("models", exist_ok=True)

    print("A: train vs biased rock")
    biased = BiasedOpponent(probs=(0.6, 0.2, 0.2))
    env_A0 = SelfPlayRPS("player_0", opponent_policy=biased, seed=args.seed)
    agent_A = train_agent(env_A0, timesteps=args.timesteps, seed=args.seed, ent=args.entropy)
    agent_A.save("models/agent_A.zip")
    wrA = evaluate_vs_opponent(agent_A, biased, episodes=300, deterministic=False)
    print(f"A: winrate vs biased: {wrA:.3f}")

    print("B: train vs frozen A")
    env_B0 = SelfPlayRPS("player_1", opponent_policy=ModelOpponent(agent_A, deterministic=False), seed=args.seed + 11)
    agent_B = train_agent(env_B0, timesteps=args.timesteps, seed=args.seed + 11, ent=args.entropy)
    agent_B.save("models/agent_B.zip")

    print("eval: A vs B (stochastic)")
    winA, draw, lossA = play_matches(agent_A, agent_B, episodes=300, deterministic=False)
    print(f"eval: A win/draw/loss: {winA:.3f} / {draw:.3f} / {lossA:.3f}")

    print("switches: alt self-play")
    env_A = SelfPlayRPS("player_0", opponent_policy=ModelOpponent(agent_B, deterministic=False), seed=args.seed + 21)
    env_B = SelfPlayRPS("player_1", opponent_policy=ModelOpponent(agent_A, deterministic=False), seed=args.seed + 31)

    agent_A.set_env(env_A)
    agent_B.set_env(env_B)

    hist = []
    for s in range(args.switches):
        if s % 2 == 0:
            print(f"switch {s+1}: train A {args.switch_steps}")
            agent_A.set_env(env_A)
            agent_A.learn(total_timesteps=args.switch_steps, progress_bar=False)
        else:
            print(f"switch {s+1}: train B {args.switch_steps}")
            agent_B.set_env(env_B)
            agent_B.learn(total_timesteps=args.switch_steps, progress_bar=False)
        wa, dr, la = play_matches(agent_A, agent_B, episodes=args.eval_episodes, deterministic=False)
        hist.append([wa, dr, la])
        print(f"  A {wa:.3f}  D {dr:.3f}  L {la:.3f}")

    np.savetxt("models/switch_history.txt", np.array(hist))
    print("saved models/switch_history.txt")

if __name__ == "__main__":
    main()