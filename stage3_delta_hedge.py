"""
Stage 3 — Delta-hedging a (hypothetical) call option on a stock
Replicates the Black-Scholes argument with the two assets you actually hold:
the stock and a risk-free account.

Position: SHORT one at-the-money European call, hedged by holding Delta shares
and keeping the rest in the risk-free account (self-financing portfolio).

  Part A  Model world (GBM): hedging error vs rebalancing frequency -> O(N^-1/2)
  Part B  Real history: sell a 3-month ATM call on every past date, price and
          hedge it with the trailing realised vol, track the final P&L.
          Theory: P&L ~ vega * (sigma_hedge - sigma_realised)
  Part C  A-share frictions: commission, stamp duty on sells, and no trading
          into a limit-up (cannot buy) or limit-down (cannot sell)

Usage (stage1_gbm.py must be in the same folder)
  python stage3_delta_hedge.py                # 600482
  python stage3_delta_hedge.py --r 0.015      # set r to your fund's yield
  python stage3_delta_hedge.py --demo
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from stage1_gbm import load_prices, estimate_gbm, TRADING_DAYS

COMMISSION = 0.00025   # per side
STAMP_DUTY = 0.0005    # sells only
LIMIT = 0.095          # |log return| above this = limit-up/down day (10% board)


# ------------------------------------------------------- Black-Scholes
def bs_call(S, K, tau, r, sigma):
    """Price and delta; handles tau = 0 (expiry)."""
    S, tau = np.asarray(S, float), np.asarray(tau, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        delta = norm.cdf(d1)
    expired = tau <= 0
    price = np.where(expired, np.maximum(S - K, 0), price)
    delta = np.where(expired, (S > K).astype(float), delta)
    return price, delta


# ------------------------------------------------------- hedging engine
def hedge_short_call(paths, K, T, r, sigma, frictions=False):
    """
    paths: (n_paths, n_steps+1) stock prices. K, sigma: scalars or (n_paths,) arrays.
    Returns final P&L per path and the history of (delta, portfolio value).
    """
    n_paths, n1 = paths.shape
    n = n1 - 1
    dt = T / n
    K = np.broadcast_to(np.asarray(K, float), (n_paths,))
    sigma = np.broadcast_to(np.asarray(sigma, float), (n_paths,))

    C0, delta = bs_call(paths[:, 0], K, T, r, sigma)
    cash = C0 - delta * paths[:, 0]            # premium in, shares bought
    if frictions:
        cash -= COMMISSION * delta * paths[:, 0]
    deltas, values = [delta.copy()], [C0.copy()]

    for i in range(1, n + 1):
        S = paths[:, i]
        cash = cash * np.exp(r * dt)           # risk-free account accrues
        values.append(cash + delta * S)        # portfolio value before rebalancing
        if i == n:
            break
        _, new_delta = bs_call(S, K, T - i * dt, r, sigma)
        trade = new_delta - delta
        if frictions:
            ret = np.log(S / paths[:, i - 1])
            blocked = ((ret > LIMIT) & (trade > 0)) | ((ret < -LIMIT) & (trade < 0))
            trade = np.where(blocked, 0.0, trade)
            cost = np.abs(trade) * S * COMMISSION + np.maximum(-trade, 0) * S * STAMP_DUTY
            cash -= cost
        cash -= trade * S
        delta = delta + trade
        deltas.append(delta.copy())

    payoff = np.maximum(paths[:, -1] - K, 0)
    pnl = values[-1] - payoff                  # what is left after paying the option holder
    return pnl, C0, np.array(deltas).T, np.array(values).T


# ------------------------------------------------------------- Part A
def part_a(S0, mu, sigma, r, T, n_paths=20000, seed=0):
    rng = np.random.default_rng(seed)
    Ns = np.array([4, 8, 16, 32, 64, 128, 256])
    stds = []
    for N in Ns:
        dt = T / N
        z = rng.standard_normal((n_paths, N))
        logp = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
        paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), logp]))
        pnl, C0, *_ = hedge_short_call(paths, S0, T, r, sigma)
        stds.append(pnl.std() / C0[0])
    slope = np.polyfit(np.log(Ns), np.log(stds), 1)[0]
    return Ns, np.array(stds), slope


# --------------------------------------------------------- Part B / C
def rolling_windows(prices, horizon):
    p = prices.values
    lookback = horizon
    r = np.diff(np.log(p))
    starts = np.arange(lookback, len(p) - horizon)
    paths = np.stack([p[s:s + horizon + 1] for s in starts])
    # vol known at the start (trailing) and vol that actually occurred (forward)
    vol_trail = np.array([r[s - lookback:s].std(ddof=1) for s in starts]) * np.sqrt(TRADING_DAYS)
    vol_real = np.array([r[s:s + horizon].std(ddof=1) for s in starts]) * np.sqrt(TRADING_DAYS)
    return prices.index[starts], paths, vol_trail, vol_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600482")
    ap.add_argument("--start", default="20211101")
    ap.add_argument("--r", type=float, default=0.015, help="risk-free rate (your fund's yield)")
    ap.add_argument("--horizon", type=int, default=60, help="option life in trading days")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    prices = load_prices(a.symbol, a.start, a.demo)
    name = prices.name
    _, mu, sigma, *_ = estimate_gbm(prices)
    T = a.horizon / TRADING_DAYS
    S_last = float(prices.iloc[-1])

    # A
    Ns, stds, slope = part_a(S_last, mu, sigma, a.r, T)

    # B, C
    dates, paths, vol_trail, vol_real = rolling_windows(prices, a.horizon)
    K = paths[:, 0]                                            # at-the-money
    pnl, C0, deltas, values = hedge_short_call(paths, K, T, a.r, vol_trail)
    pnl_f, *_ = hedge_short_call(paths, K, T, a.r, vol_trail, frictions=True)
    rel, rel_f = pnl / C0, pnl_f / C0
    gap = vol_trail - vol_real
    beta, alpha = np.polyfit(gap, rel, 1)
    corr = np.corrcoef(gap, rel)[0, 1]
    # benchmark: same windows, but hedged with the vol that actually occurred (perfect foresight)
    pnl_o, C0_o, *_ = hedge_short_call(paths, K, T, a.r, vol_real)
    rel_o = pnl_o / C0_o

    print(f"\n=== {name}: short {a.horizon}-day ATM call, daily delta hedge, r = {a.r:.2%} ===")
    print(f"[A] GBM world: hedging-error std vs N has slope {slope:.2f} (theory -0.50); "
          f"daily hedging -> std = {np.interp(a.horizon, Ns, stds):.1%} of premium")
    print(f"[B] {len(rel)} overlapping windows on real history, hedge vol = trailing {a.horizon}-day vol")
    print(f"    P&L / premium: mean {rel.mean():+.1%}, std {rel.std():.1%}, "
          f"5% worst {np.percentile(rel, 5):+.1%}, best 95% {np.percentile(rel, 95):+.1%}")
    print(f"    corr(P&L, sigma_hedge - sigma_realised) = {corr:.2f}; "
          f"1 vol point of gap = {beta/100:+.1%} of premium")
    print(f"    perfect-foresight vol (hedge with realised): std {rel_o.std():.1%} "
          f"-> the rest of the risk is vol-forecast error")
    print(f"[C] with A-share frictions: mean {rel_f.mean():+.1%} "
          f"(cost of frictions = {rel.mean() - rel_f.mean():.1%} of premium), std {rel_f.std():.1%}")
    print("    note: overlapping windows are highly correlated; treat these as descriptive, not as t-stats")

    # ---- plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    w = -1                                                      # most recent complete window
    t = np.arange(a.horizon + 1)
    opt_val, _ = bs_call(paths[w], K[w], T - t / TRADING_DAYS, a.r, vol_trail[w])

    ax[0, 0].plot(t, paths[w], label="stock")
    ax[0, 0].axhline(K[w], color="k", ls=":", label="strike")
    ax2 = ax[0, 0].twinx()
    ax2.step(t[:-1], deltas[w], where="post", color="tab:red", alpha=0.7)
    ax2.set_ylabel("delta (shares held)", color="tab:red"); ax2.set_ylim(0, 1)
    ax[0, 0].set_title(f"Latest window (from {dates[w].date()}): stock and hedge ratio")
    ax[0, 0].set_xlabel("trading days"); ax[0, 0].legend(fontsize=8, loc="upper left")

    ax[0, 1].plot(t, opt_val, label="option value (liability)")
    ax[0, 1].plot(t, values[w], "--", label="replicating portfolio")
    ax[0, 1].set_title(f"Replication: final P&L = {rel[w]:+.0%} of premium")
    ax[0, 1].set_xlabel("trading days"); ax[0, 1].legend(fontsize=8)

    ax[0, 2].loglog(Ns, stds, "o-", label=f"simulated, slope = {slope:.2f}")
    ax[0, 2].loglog(Ns, stds[0] * (Ns / Ns[0]) ** -0.5, "k:", label="reference slope -0.5")
    ax[0, 2].set_title("GBM world: hedging error vs rebalancing frequency")
    ax[0, 2].set_xlabel("rebalances over option life"); ax[0, 2].set_ylabel("std(P&L) / premium")
    ax[0, 2].legend(fontsize=8)

    ax[1, 0].scatter(gap, rel, s=5, alpha=0.4)
    xs = np.linspace(gap.min(), gap.max(), 50)
    ax[1, 0].plot(xs, alpha + beta * xs, "r", label=f"fit, corr = {corr:.2f}")
    ax[1, 0].axhline(0, color="k", lw=0.5); ax[1, 0].axvline(0, color="k", lw=0.5)
    ax[1, 0].set_title("Real history: P&L is a bet on volatility, not direction")
    ax[1, 0].set_xlabel("$\\sigma_{hedge} - \\sigma_{realised}$"); ax[1, 0].set_ylabel("P&L / premium")
    ax[1, 0].legend(fontsize=8)

    bins = np.linspace(min(rel.min(), rel_f.min()), max(rel.max(), rel_f.max()), 60)
    ax[1, 1].hist(rel, bins=bins, alpha=0.6, label=f"frictionless (mean {rel.mean():+.0%})")
    ax[1, 1].hist(rel_f, bins=bins, alpha=0.6, label=f"A-share frictions (mean {rel_f.mean():+.0%})")
    ax[1, 1].set_title("P&L distribution across all start dates")
    ax[1, 1].set_xlabel("P&L / premium"); ax[1, 1].legend(fontsize=8)

    ax[1, 2].plot(dates, vol_trail, label="vol used to price & hedge (trailing)")
    ax[1, 2].plot(dates, vol_real, label="vol realised over option life")
    ax[1, 2].fill_between(dates, vol_trail, vol_real, where=vol_real > vol_trail,
                          color="tab:red", alpha=0.25, label="seller loses")
    ax[1, 2].set_title("Why it loses: realised vol jumps above the hedge vol")
    ax[1, 2].tick_params(axis="x", rotation=30); ax[1, 2].legend(fontsize=8)

    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    fig.suptitle(f"Stage 3 — Delta-hedging a short {a.horizon}-day ATM call on {name}", fontsize=13)
    fig.tight_layout()
    out = f"stage3_delta_hedge_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
