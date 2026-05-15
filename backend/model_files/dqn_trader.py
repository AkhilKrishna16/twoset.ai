"""Deep Q-Learning stock trading agent built from scratch.

The agent learns a discrete trading policy (hold / buy / sell) over a single
asset. It reuses the GRU sequence-encoder idea from the project's
bidirectional-gru forecaster, but instead of predicting the next price the
encoder feeds into a Q-value head that scores each available action.

The environment follows an OpenEnv-style contract: `reset()` and `step(action)`
both return a typed `Observation` dataclass with `obs`, `reward`, `done`, and
an `info` dict. That makes the env a drop-in target for the OpenEnv finance
runner once a server wrapper is added.

Run directly:

    python -m backend.model_files.dqn_trader                  # single ticker (AAPL)
    python -m backend.model_files.dqn_trader --tickers mag7   # train across Mag-7

In single-ticker mode it downloads AAPL via yfinance, trains a DQN on the
first 80% of history, and evaluates greedily on the held-out 20%. In
multi-ticker mode it trains one shared agent across all supplied tickers
(round-robin per episode) and reports per-ticker plus aggregate profit on
each held-out split. Target: >= $2,000 profit per $10,000 bankroll.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
TICKER_GROUPS = {"mag7": MAG7}


def parse_tickers(spec: str) -> list[str]:
    """Resolve a comma-separated ticker spec or a group alias (e.g. "mag7")."""
    spec = spec.strip()
    if spec.lower() in TICKER_GROUPS:
        return list(TICKER_GROUPS[spec.lower()])
    return [t.strip().upper() for t in spec.split(",") if t.strip()]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# OpenEnv-style trading environment
# ---------------------------------------------------------------------------

WINDOW = 30
INITIAL_CASH = 10_000.0
COMMISSION = 1e-4  # 1 bp per trade


@dataclass
class Observation:
    """OpenEnv-style step return.

    `obs` is the agent-facing feature vector. `reward`, `done`, `info` mirror
    the conventions used by OpenEnv environment servers so an HTTP wrapper
    can serialize this directly.
    """

    obs: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class StockTradingEnv:
    """Single-asset trading environment.

    Actions
    -------
    0 = HOLD: keep current cash/shares mix.
    1 = BUY:  convert all cash into shares at the current close.
    2 = SELL: convert all shares back to cash at the current close.

    Observation
    -----------
    Concatenation of `WINDOW` recent log returns (rolling) plus two
    portfolio-state scalars: cash ratio and position ratio.

    Reward
    ------
    Log return of the mark-to-market portfolio value step-over-step. Log
    returns sum to the total log return over an episode, so maximising
    cumulative reward maximises terminal wealth.
    """

    action_space = 3

    def __init__(
        self,
        prices: np.ndarray,
        window: int = WINDOW,
        initial_cash: float = INITIAL_CASH,
        commission: float = COMMISSION,
    ) -> None:
        if len(prices) <= window + 1:
            raise ValueError("Need more prices than the window length")
        self.prices = prices.astype(np.float32)
        self.window = window
        self.initial_cash = initial_cash
        self.commission = commission
        self.obs_dim = window + 2

    def reset(self) -> Observation:
        self.t = self.window
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_value = self._mark_to_market()
        self.starting_value = self.portfolio_value
        return Observation(
            obs=self._build_obs(),
            reward=0.0,
            done=False,
            info={"portfolio_value": self.portfolio_value},
        )

    def step(self, action: int) -> Observation:
        price = float(self.prices[self.t])
        if action == 1 and self.cash > 0:
            self.shares += (self.cash * (1.0 - self.commission)) / price
            self.cash = 0.0
        elif action == 2 and self.shares > 0:
            self.cash += self.shares * price * (1.0 - self.commission)
            self.shares = 0.0

        prev_value = self.portfolio_value
        self.t += 1
        done = self.t >= len(self.prices) - 1
        self.portfolio_value = self._mark_to_market()
        reward = math.log((self.portfolio_value + 1e-8) / (prev_value + 1e-8))
        obs = self._build_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "portfolio_value": self.portfolio_value,
            "price": price,
            "action": action,
        }
        return Observation(obs=obs, reward=reward, done=done, info=info)

    def _mark_to_market(self) -> float:
        return self.cash + self.shares * float(self.prices[self.t])

    def _build_obs(self) -> np.ndarray:
        window_prices = self.prices[self.t - self.window : self.t]
        # Prices are strictly positive (yfinance closes); no epsilon needed.
        log_prices = np.log(window_prices)
        log_returns = np.diff(log_prices, prepend=log_prices[0]).astype(np.float32)
        port = self._mark_to_market() + 1e-8
        cash_ratio = self.cash / port
        pos_ratio = (self.shares * float(self.prices[self.t])) / port
        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[: self.window] = log_returns
        obs[self.window] = cash_ratio
        obs[self.window + 1] = pos_ratio
        return obs


# ---------------------------------------------------------------------------
# Deep Q-Network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """GRU encoder over the price window + MLP head producing Q-values.

    The encoder mirrors the BiGRU forecaster used elsewhere in the project,
    but emits Q-values for the three trading actions instead of a next-price
    estimate.
    """

    def __init__(self, window: int, n_aux: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.window = window
        self.n_aux = n_aux
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_aux, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = x[:, : self.window].unsqueeze(-1)
        aux = x[:, self.window :]
        _, h = self.gru(window)
        return self.head(torch.cat([h[-1], aux], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffer + agent
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-size ring buffer backed by pre-allocated numpy arrays.

    Storing transitions in contiguous typed arrays avoids the per-sample
    Python tuple + `np.array(list_of_arrays)` cost in the hot training loop
    and keeps dtypes stable (float32 / int64) for zero-copy transfer to
    torch tensors.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, s, a, r, ns, d) -> None:
        i = self.idx
        self.states[i] = s
        self.next_states[i] = ns
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = d
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_states[idxs]),
            torch.from_numpy(self.dones[idxs]),
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """Double-DQN with soft target updates and an epsilon-greedy explorer."""

    def __init__(
        self,
        obs_dim: int,
        window: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.92,
        tau: float = 0.01,
        batch_size: int = 128,
        buffer_capacity: int = 50_000,
        reward_scale: float = 100.0,
    ) -> None:
        self.window = window
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        n_aux = obs_dim - window
        self.policy = QNetwork(window, n_aux, n_actions).to(DEVICE)
        self.target = QNetwork(window, n_aux, n_actions).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        # Target network is only ever consulted under no_grad; mark params
        # accordingly so autograd doesn't track them and BN/Dropout (if added
        # later) stay in inference mode.
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            return int(self.policy(s).argmax(dim=-1).item())

    def update(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, 1000):
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = s.to(DEVICE, non_blocking=True)
        a = a.to(DEVICE, non_blocking=True)
        r = r.to(DEVICE, non_blocking=True) * self.reward_scale
        ns = ns.to(DEVICE, non_blocking=True)
        d = d.to(DEVICE, non_blocking=True)
        q_sa = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.policy(ns).argmax(dim=-1)
            next_q = self.target(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * (1.0 - d) * next_q
        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optim.step()
        # Polyak (soft) update: tp <- (1 - tau) * tp + tau * p, vectorized
        # over the parameter list to avoid Python-loop overhead per param.
        with torch.no_grad():
            target_params = [tp.data for tp in self.target.parameters()]
            policy_params = [p.data for p in self.policy.parameters()]
            torch._foreach_mul_(target_params, 1.0 - self.tau)
            torch._foreach_add_(target_params, policy_params, alpha=self.tau)
        return float(loss.item())

    def decay_eps(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------


def load_prices(ticker: str = "AAPL", period: str = "10y") -> np.ndarray:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    # yfinance may return a DataFrame (multi-ticker / MultiIndex columns) or
    # a Series depending on version; squeeze handles both.
    prices = np.asarray(close.to_numpy()).squeeze().astype(np.float32)
    prices = prices[np.isfinite(prices)]
    if prices.ndim != 1 or prices.size == 0:
        raise RuntimeError(f"Could not extract a 1D close series for {ticker}")
    return prices


def run_episode(
    env: StockTradingEnv, agent: DQNAgent, train: bool, record: bool = False
) -> dict:
    obs = env.reset()
    state = obs.obs
    total_reward = 0.0
    update_every = 4
    step = 0
    trajectory: list[dict] = []
    if record:
        trajectory.append(
            {
                "t": env.t,
                "price": float(env.prices[env.t]),
                "action": -1,
                "portfolio_value": env.portfolio_value,
            }
        )
    while True:
        action = agent.act(state, greedy=not train)
        out = env.step(action)
        if train:
            agent.buffer.push(state, action, out.reward, out.obs, float(out.done))
            if step % update_every == 0:
                agent.update()
        if record:
            trajectory.append(
                {
                    "t": env.t,
                    "price": float(out.info["price"]),
                    "action": int(action),
                    "portfolio_value": float(out.info["portfolio_value"]),
                }
            )
        state = out.obs
        total_reward += out.reward
        step += 1
        if out.done:
            break
    return {
        "log_return": total_reward,
        "profit": env.portfolio_value - env.starting_value,
        "final_value": env.portfolio_value,
        "trajectory": trajectory,
    }


def plot_run(
    trajectory: list[dict],
    ticker: str,
    out_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    steps = [r["t"] for r in trajectory]
    prices = [r["price"] for r in trajectory]
    port = [r["portfolio_value"] for r in trajectory]
    actions = np.array([r["action"] for r in trajectory])
    buys = [(s, p) for s, p, a in zip(steps, prices, actions) if a == 1]
    sells = [(s, p) for s, p, a in zip(steps, prices, actions) if a == 2]

    buy_hold_curve = (np.array(prices) / prices[0]) * INITIAL_CASH

    fig, (ax_price, ax_port) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_price.plot(steps, prices, label=f"{ticker} close", color="steelblue")
    if buys:
        bx, by = zip(*buys)
        ax_price.scatter(bx, by, marker="^", color="green", s=60, label="BUY", zorder=3)
    if sells:
        sx, sy = zip(*sells)
        ax_price.scatter(sx, sy, marker="v", color="red", s=60, label="SELL", zorder=3)
    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(f"{ticker} — DQN actions on held-out test split")
    ax_price.legend(loc="best")
    ax_price.grid(alpha=0.3)

    ax_port.plot(steps, port, label="DQN portfolio", color="darkorange")
    ax_port.plot(steps, buy_hold_curve, label="Buy & hold", color="gray", linestyle="--")
    ax_port.axhline(INITIAL_CASH, color="black", linewidth=0.8, alpha=0.5)
    ax_port.set_xlabel("Test step (day index)")
    ax_port.set_ylabel("Portfolio value ($)")
    ax_port.legend(loc="best")
    ax_port.grid(alpha=0.3)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=130)
        print(f"[plot] saved -> {out_path}")
    else:
        plt.show()


def main(
    num_episodes: int = 25,
    tickers: list[str] | None = None,
    period: str = "10y",
    plot: bool = False,
    plot_path: Path | None = None,
) -> dict:
    seed_everything()
    tickers = tickers or ["AAPL"]

    train_envs: dict[str, StockTradingEnv] = {}
    test_envs: dict[str, StockTradingEnv] = {}
    test_prices_by_ticker: dict[str, np.ndarray] = {}
    for tk in tickers:
        prices = load_prices(tk, period=period)
        split = int(0.8 * len(prices))
        train_prices, test_prices = prices[:split], prices[split:]
        if len(test_prices) <= WINDOW + 1:
            print(f"[data] {tk}: test split too short ({len(test_prices)} days); skipping")
            continue
        train_envs[tk] = StockTradingEnv(train_prices)
        test_envs[tk] = StockTradingEnv(test_prices)
        test_prices_by_ticker[tk] = test_prices
        print(
            f"[data] {tk:5s} train_days={len(train_prices)} test_days={len(test_prices)} "
            f"test_range=${test_prices.min():.2f}-${test_prices.max():.2f}"
        )

    if not train_envs:
        raise RuntimeError("No usable tickers after data loading.")

    sample_env = next(iter(train_envs.values()))
    agent = DQNAgent(
        obs_dim=sample_env.obs_dim,
        window=WINDOW,
        n_actions=sample_env.action_space,
    )

    best_aggregate_test = -float("inf")
    for ep in range(1, num_episodes + 1):
        order = list(train_envs.keys())
        random.shuffle(order)
        train_profits: dict[str, float] = {}
        for tk in order:
            train_profits[tk] = run_episode(train_envs[tk], agent, train=True)["profit"]
        agent.decay_eps()

        test_profits = {
            tk: run_episode(env, agent, train=False)["profit"]
            for tk, env in test_envs.items()
        }
        agg_test = sum(test_profits.values())
        best_aggregate_test = max(best_aggregate_test, agg_test)
        per_ticker = " ".join(f"{tk}=${p:+,.0f}" for tk, p in test_profits.items())
        print(
            f"[ep {ep:02d}] eps={agent.eps:.3f}  "
            f"test_agg=${agg_test:+,.2f}  best_agg=${best_aggregate_test:+,.2f}  "
            f"[{per_ticker}]"
        )

    final_results: dict[str, dict] = {}
    total_profit = 0.0
    total_buy_hold = 0.0
    print("")
    for tk, env in test_envs.items():
        run = run_episode(env, agent, train=False, record=plot)
        test_prices = test_prices_by_ticker[tk]
        buy_hold = (test_prices[-1] / test_prices[WINDOW]) * INITIAL_CASH - INITIAL_CASH
        final_results[tk] = {
            "profit": run["profit"],
            "buy_hold": float(buy_hold),
            "trajectory": run["trajectory"],
        }
        total_profit += run["profit"]
        total_buy_hold += float(buy_hold)
        print(
            f"[result] {tk:5s} DQN=${run['profit']:+,.2f}  "
            f"buy&hold=${buy_hold:+,.2f}  "
            f"{'PASS' if run['profit'] >= 2000 else 'FAIL'}"
        )
    n = len(test_envs)
    print("")
    print(f"[result] aggregate DQN profit: ${total_profit:+,.2f}  (avg ${total_profit / n:+,.2f}/ticker)")
    print(f"[result] aggregate buy&hold:   ${total_buy_hold:+,.2f}  (avg ${total_buy_hold / n:+,.2f}/ticker)")

    if plot:
        for tk, res in final_results.items():
            out = plot_path
            if out is not None and len(final_results) > 1:
                out = plot_path.with_name(f"{plot_path.stem}_{tk}{plot_path.suffix}")
            plot_run(res["trajectory"], ticker=tk, out_path=out)

    return {
        "per_ticker": {tk: res["profit"] for tk, res in final_results.items()},
        "aggregate_profit": total_profit,
        "aggregate_buy_hold": total_buy_hold,
        "best_aggregate_test_profit": best_aggregate_test,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train + evaluate a DQN trading agent.")
    parser.add_argument("--ticker", default=None, help="Single ticker (e.g. AAPL). Ignored if --tickers is set.")
    parser.add_argument(
        "--tickers",
        default=None,
        help='Comma-separated tickers or a group alias (e.g. "AAPL,MSFT" or "mag7").',
    )
    parser.add_argument("--episodes", type=int, default=25, help="Number of training episodes.")
    parser.add_argument("--period", default="10y", help="yfinance history period (e.g. 5y, 10y, max).")
    parser.add_argument("--plot", action="store_true", help="Show actions and portfolio vs price on the test split.")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="If set with --plot, save the figure to this path. With multiple tickers, the ticker is appended.",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = parse_tickers(args.tickers)
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = ["AAPL"]

    main(
        num_episodes=args.episodes,
        tickers=tickers,
        period=args.period,
        plot=args.plot,
        plot_path=args.plot_path,
    )