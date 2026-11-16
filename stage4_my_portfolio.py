"""
Stage 4 — Applying the models to an actual two-asset portfolio
(N shares of the stock + a cash amount in a risk-free fund)

  Part A  Risk snapshot: weight, portfolio vol, 1-day VaR (normal vs historical),
          1-year wealth distribution (GBM vs bootstrap of real returns)
  Part B  Merton's portfolio problem: optimal stock weight pi* = (mu - r) / (gamma sigma^2),
          its sensitivity to the (unknowable) mu, and what your actual weight implies
  Part C  Synthetic protective put: no listed options on the stock, so replicate
          "shares + put" by shifting money between the stock and the fund
          (hold N * N(d1) shares), in 100-share lots, on real history

Usage (stage1_gbm.py must be in the same folder)
  python stage4_my_portfolio.py --shares 2000 --cash 200000 --r 0.011 --price 28.5
  python stage4_my_portfolio.py --demo
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from stage1_gbm import load_prices, estimate_gbm, TRADING_DAYS

LOT = 100              # A-share board lot
COMMISSION, STAMP_DUTY = 0.00025, 0.0005


def d1(S, K, tau, r, sigma):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return x


def bs_put(S, K, T, r, sigma):
    a = d1(S, K, T, r, sigma)
    b = a - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-b) - S * norm.cdf(-a)


def synthetic_put(paths, n_shares, cash, T, r, sigma, lot=LOT):
    """Hold n*N(d1) shares (strike = starting price), rest in the fund. Returns final wealth."""
    W, n1 = paths.shape
    n, dt = n1 - 1, T / (n1 - 1)
    K = paths[:, 0]
    target = lambda S, tau: n_shares * norm.cdf(d1(S, K, tau, r, sigma))
    rnd = (lambda x: np.round(x / lot) * lot) if lot else (lambda x: x)
    h = rnd(target(paths[:, 0], T))
    fund = cash + (n_shares - h) * paths[:, 0] * (1 - COMMISSION - STAMP_DUTY)   # initial sale
    for i in range(1, n):
        S = paths[:, i]
        fund *= np.exp(r * dt)
        new_h = rnd(target(S, T - i * dt))
        trade = new_h - h
        fund -= trade * S + np.abs(trade) * S * COMMISSION + np.maximum(-trade, 0) * S * STAMP_DUTY
        h = new_h
    fund *= np.exp(r * dt)
    return fund + h * paths[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600482"); ap.add_argument("--start", default="20211101")
    ap.add_argument("--shares", type=int, default=2000)
    ap.add_argument("--cash", type=float, default=200000)
    ap.add_argument("--r", type=float, default=0.011)
    ap.add_argument("--price", type=float, help="current quoted price (default: last price in the data)")
    ap.add_argument("--gamma", type=float, default=3.0, help="relative risk aversion for Part B")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    prices = load_prices(a.symbol, a.start, a.demo)
    ret, mu, sigma, se_mu, _ = estimate_gbm(prices)
    S0 = a.price or float(prices.iloc[-1])
    stock, total = a.shares * S0, a.shares * S0 + a.cash
    w = stock / total
    rng = np.random.default_rng(0)

    # ---------------- A
    var_n = stock * 2.326 * sigma / np.sqrt(TRADING_DAYS)
    var_h = -stock * (np.exp(np.percentile(ret, 1)) - 1)
    z = rng.standard_normal(100000)
    gbm_1y = np.exp((mu - 0.5 * sigma**2) + sigma * z)
    boot_1y = np.exp(rng.choice(ret.values, (100000, TRADING_DAYS)).sum(1))
    wealth = lambda g: a.cash * np.exp(a.r) + stock * g
    print(f"\n=== Portfolio: {a.shares} shares x {S0:.2f} = {stock:,.0f}  +  fund {a.cash:,.0f}  =  {total:,.0f} ===")
    print(f"[A] stock weight {w:.1%}; portfolio vol = {w*sigma:.1%} (stock alone {sigma:.1%})")
    print(f"    1-day 99% VaR: normal {var_n:,.0f}   historical {var_h:,.0f}   (limit-down day: {stock*0.10:,.0f})")
    for lab, g in (("GBM", gbm_1y), ("bootstrap", boot_1y)):
        q = np.percentile(wealth(g), [5, 50, 95]) / total - 1
        print(f"    1-year total return, {lab:9s}: 5% {q[0]:+.1%}   median {q[1]:+.1%}   95% {q[2]:+.1%}")
    print(f"    fund-only benchmark: {np.exp(a.r)-1:+.2%}")

    # ---------------- B
    pi = lambda m, g: (m - a.r) / (g * sigma**2)
    print(f"[B] Merton weight at gamma = {a.gamma:g}: {pi(mu, a.gamma):.0%}  "
          f"(mu ± 1 s.e.: {pi(mu - se_mu, a.gamma):.0%} to {pi(mu + se_mu, a.gamma):.0%})")
    print(f"    your {w:.0%} is optimal if gamma = {(mu - a.r)/(w*sigma**2):.1f} (taking mu-hat at face value)")
    print(f"    or, at gamma = {a.gamma:g}, if you believe mu = {a.r + a.gamma*w*sigma**2:.1%}  "
          f"(vs mu-hat = {mu:.1%} ± {se_mu:.1%})")

    # ---------------- C
    p = prices.values
    r_log = np.diff(np.log(p))
    H, T = a.horizon, a.horizon / TRADING_DAYS
    starts = np.arange(H, len(p) - H)
    paths = np.stack([p[s:s + H + 1] / p[s] * S0 for s in starts])     # rescaled to today's price
    vol = np.array([r_log[s - H:s].std(ddof=1) for s in starts]) * np.sqrt(TRADING_DAYS)
    ST = paths[:, -1]
    unhedged = a.shares * ST + a.cash * np.exp(a.r * T)
    premium = a.shares * bs_put(S0, S0, T, a.r, vol)
    ideal = a.shares * np.maximum(ST, S0) + (a.cash - premium) * np.exp(a.r * T)
    syn = synthetic_put(paths, a.shares, a.cash, T, a.r, vol)
    syn_cont = synthetic_put(paths, a.shares, a.cash, T, a.r, vol, lot=0)
    R = lambda x: x / total - 1
    print(f"[C] synthetic protective put on {a.shares} shares, {H}-day horizon, {len(starts)} historical windows")
    print(f"    put premium a real option would cost: median {np.median(premium):,.0f} "
          f"({np.median(premium)/stock:.1%} of the stock position)")
    print(f"    {'':22s}{'worst':>9s}{'5%':>9s}{'median':>9s}{'95%':>9s}")
    for lab, x in (("unhedged", unhedged), ("synthetic put", syn), ("ideal listed put", ideal)):
        q = np.percentile(R(x), [0, 5, 50, 95])
        print(f"    {lab:22s}{q[0]:+9.1%}{q[1]:+9.1%}{q[2]:+9.1%}{q[3]:+9.1%}")
    print(f"    replication error (synthetic - ideal): std {np.std(syn - ideal):,.0f}; "
          f"with fractional shares {np.std(syn_cont - ideal):,.0f} -> lot-size effect")

    # ---------------- plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    name = prices.name
    bins = np.linspace(-0.25, 0.45, 90)
    ax[0, 0].hist(R(wealth(gbm_1y)), bins=bins, density=True, alpha=0.6, label="GBM")
    ax[0, 0].hist(R(wealth(boot_1y)), bins=bins, density=True, alpha=0.6, label="bootstrap of real returns")
    ax[0, 0].axvline(np.exp(a.r) - 1, color="k", ls=":", label="fund only")
    ax[0, 0].set_title(f"1-year return of the whole portfolio ({w:.0%} in stock)")
    ax[0, 0].set_xlabel("total return"); ax[0, 0].legend(fontsize=8)

    ws = np.linspace(0, 1, 101)
    q05 = [np.percentile(wi * gbm_1y + (1 - wi) * np.exp(a.r), 5) - 1 for wi in ws]
    q50 = [np.percentile(wi * gbm_1y + (1 - wi) * np.exp(a.r), 50) - 1 for wi in ws]
    ax[0, 1].plot(ws, q50, label="median 1-year return")
    ax[0, 1].plot(ws, q05, label="5th percentile")
    ax[0, 1].axvline(w, color="r", ls="--", label=f"your weight {w:.0%}")
    ax[0, 1].set_title("Risk vs stock weight (GBM, taking $\\hat\\mu$ at face value)")
    ax[0, 1].set_xlabel("weight in stock"); ax[0, 1].legend(fontsize=8)

    gs = np.linspace(1, 10, 100)
    ax[0, 2].plot(gs, pi(mu, gs), "k", label="$\\pi^*$ at $\\hat\\mu$")
    ax[0, 2].fill_between(gs, pi(mu - se_mu, gs), pi(mu + se_mu, gs), alpha=0.25, label="$\\hat\\mu$ ± 1 s.e.")
    ax[0, 2].axhline(w, color="r", ls="--", label=f"your weight {w:.0%}")
    ax[0, 2].set_ylim(-0.5, 2); ax[0, 2].set_title("Merton weight: the answer is mostly estimation error")
    ax[0, 2].set_xlabel("risk aversion $\\gamma$"); ax[0, 2].set_ylabel("optimal stock weight")
    ax[0, 2].legend(fontsize=8)

    x = ST / S0 - 1
    o = np.argsort(x)
    ax[1, 0].plot(x[o], R(unhedged)[o], "k", lw=1, label="unhedged")
    ax[1, 0].scatter(x, R(syn), s=5, alpha=0.4, color="tab:orange", label="synthetic put (100-share lots)")
    ax[1, 0].scatter(x, R(ideal), s=3, alpha=0.3, color="tab:green", label="ideal listed put")
    ax[1, 0].set_title(f"{H}-day outcome of the whole portfolio vs stock move")
    ax[1, 0].set_xlabel("stock return over the window"); ax[1, 0].set_ylabel("portfolio return")
    ax[1, 0].legend(fontsize=8)

    b2 = np.linspace(min(R(unhedged).min(), R(syn).min()), max(R(unhedged).max(), R(syn).max()), 60)
    ax[1, 1].hist(R(unhedged), bins=b2, alpha=0.6, label="unhedged")
    ax[1, 1].hist(R(syn), bins=b2, alpha=0.6, label="synthetic put")
    ax[1, 1].set_title("Protection trims the left tail and costs upside")
    ax[1, 1].set_xlabel("portfolio return"); ax[1, 1].legend(fontsize=8)

    k = -1
    path = paths[k]
    tau = T - np.arange(H) / TRADING_DAYS
    h_exact = a.shares * norm.cdf(d1(path[:-1], path[0], tau, a.r, vol[k]))
    ax[1, 2].step(range(H), np.round(h_exact / LOT) * LOT, where="post", label="shares held (lots)")
    ax[1, 2].plot(range(H), h_exact, ":", label="exact $N\\cdot N(d_1)$")
    ax2 = ax[1, 2].twinx(); ax2.plot(range(H + 1), path, color="gray", alpha=0.6); ax2.set_ylabel("stock price")
    ax[1, 2].set_title("Latest window: shares to hold day by day")
    ax[1, 2].set_xlabel("trading days"); ax[1, 2].legend(fontsize=8, loc="upper left")

    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    fig.suptitle(f"Stage 4 — {a.shares} shares of {name} + {a.cash:,.0f} in a risk-free fund (r = {a.r:.2%})", fontsize=13)
    fig.tight_layout()
    out = f"stage4_my_portfolio_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
