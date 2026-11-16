"""
Stage 2 — Black-Scholes pricing, Greeks and implied volatility

  Part A  BSM pricer + analytic Greeks, checked against finite differences
  Part B  What the Greeks look like for an option on a 600482-like stock
          (sigma from Stage 1)
  Part C  Implied-vol smile from real SSE ETF options (A-shares have no
          single-stock options). The forward and discount factor are backed
          out of put-call parity, so no spot price or dividend guess is needed.

Usage
  pip install akshare numpy pandas scipy matplotlib
  python stage2_bsm_iv.py                         # 50ETF options, next two expiries
  python stage2_bsm_iv.py --underlying 300ETF
  python stage2_bsm_iv.py --csv chain.csv --expiry 2026-12-23
        # your own chain; columns: type (C/P), strike, price
  python stage2_bsm_iv.py --demo                  # no network needed
"""
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

BOARD = {"50ETF": "华夏上证50ETF期权", "300ETF": "华泰柏瑞沪深300ETF期权", "500ETF": "南方中证500ETF期权"}


# ------------------------------------------------------- pricing core
def black(F, K, T, sigma, D, cp=1):
    """Black-76 price of a European option on forward F. cp=+1 call, -1 put. D = discount factor."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return D * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def bsm(S, K, T, r, sigma, cp=1):
    """Black-Scholes-Merton price and Greeks for a non-dividend stock."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    return {
        "price": cp * (S * norm.cdf(cp * d1) - K * disc * norm.cdf(cp * d2)),
        "delta": cp * norm.cdf(cp * d1),
        "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        "vega": S * norm.pdf(d1) * np.sqrt(T),
        "theta": -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - cp * r * K * disc * norm.cdf(cp * d2),
    }


def implied_vol(price, F, K, T, D, cp):
    """Invert Black-76 with a bracketing root finder; NaN if the price violates no-arbitrage bounds."""
    intrinsic = D * max(cp * (F - K), 0.0)
    if not (intrinsic < price < D * (F if cp == 1 else K)):
        return np.nan
    try:
        return brentq(lambda s: black(F, K, T, s, D, cp) - price, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


# ------------------------------------------------------------- Part A
def check_greeks(S=100.0, K=105.0, T=0.25, r=0.015, sigma=0.358):
    g, h = bsm(S, K, T, r, sigma), 1e-3
    p = lambda **kw: bsm(**{**dict(S=S, K=K, T=T, r=r, sigma=sigma), **kw})["price"]
    fd = {
        "delta": (p(S=S + h) - p(S=S - h)) / (2 * h),
        "gamma": (p(S=S + h) - 2 * p() + p(S=S - h)) / h**2,
        "vega": (p(sigma=sigma + h) - p(sigma=sigma - h)) / (2 * h),
        "theta": -(p(T=T + h) - p(T=T - h)) / (2 * h),
    }
    print("[A] analytic Greeks vs finite differences (call, S=100, K=105, T=3m)")
    for k in fd:
        print(f"    {k:5s}  analytic {g[k]:+10.5f}   finite-diff {fd[k]:+10.5f}   diff {abs(g[k]-fd[k]):.1e}")
    # the BS PDE must hold: theta + r S delta + 0.5 sigma^2 S^2 gamma - r V = 0
    pde = g["theta"] + r * S * g["delta"] + 0.5 * sigma**2 * S**2 * g["gamma"] - r * g["price"]
    print(f"    Black-Scholes PDE residual: {pde:.1e}")
    iv = implied_vol(g["price"], S * np.exp(r * T), K, T, np.exp(-r * T), 1)
    print(f"    implied-vol round trip: input {sigma:.4f} -> recovered {iv:.4f}")


# ------------------------------------------------------------- Part C
def fourth_wednesday(year, month):
    d = dt.date(year, month, 1)
    first_wed = d + dt.timedelta(days=(2 - d.weekday()) % 7)
    return first_wed + dt.timedelta(days=21)


def next_expiries(today, n=2):
    out, y, m = [], today.year, today.month
    while len(out) < n:
        e = fourth_wednesday(y, m)
        if (e - today).days >= 5:              # skip an expiry that is too close
            out.append(e)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_chain(underlying, expiry):
    """Returns DataFrame[type, strike, price] for one expiry from akshare."""
    import akshare as ak
    df = ak.option_finance_board(symbol=BOARD[underlying], end_month=expiry.strftime("%y%m"))
    code = df["合约交易代码"].astype(str)
    out = pd.DataFrame({
        "type": code.str[6],                    # 510050C2612M03000 -> 'C'
        "strike": pd.to_numeric(df["行权价"], errors="coerce"),
        "price": pd.to_numeric(df["当前价"], errors="coerce"),
    })
    out = out[code.str[11] == "M"]              # drop dividend-adjusted contracts (A/B)
    return out.dropna()


def demo_chain(T, F=3.0, r=0.015, seed=0):
    rng = np.random.default_rng(seed)
    D, rows = np.exp(-r * T), []
    for K in np.round(np.arange(2.6, 3.45, 0.05), 2):
        k = np.log(K / F)
        vol = 0.17 - 0.25 * k + 1.8 * k**2 + 0.01 / np.sqrt(T) * abs(k) * 4   # skew + curvature
        for cp, t in ((1, "C"), (-1, "P")):
            px = black(F, K, T, vol, D, cp) + rng.normal(0, 1e-4)
            rows.append((t, K, round(max(px, 1e-4), 4)))
    return pd.DataFrame(rows, columns=["type", "strike", "price"])


def parity_forward(chain):
    """C - P = D (F - K): regress (C - P) on K over near-the-money strikes."""
    w = chain.pivot_table(index="strike", columns="type", values="price").dropna()
    y = (w["C"] - w["P"])
    atm = y.abs().sort_values().index[:7]       # strikes closest to the forward
    slope, intercept = np.polyfit(atm.values, y.loc[atm].values, 1)
    D = -slope
    return intercept / D, D, w


def smile(chain, T):
    F, D, w = parity_forward(chain)
    rows = []
    for K, row in w.iterrows():
        cp = 1 if K >= F else -1                # use the out-of-the-money side: more liquid, more vega
        px = row["C"] if cp == 1 else row["P"]
        if px < 0.001:                          # quotes at the minimum tick carry no vol information
            continue
        rows.append((K, K / F, px, cp, implied_vol(px, F, K, T, D, cp)))
    s = pd.DataFrame(rows, columns=["strike", "moneyness", "price", "cp", "iv"]).dropna()
    atm_iv = float(np.interp(1.0, s["moneyness"], s["iv"]))
    s["flat_price"] = [black(F, k, T, atm_iv, D, c) for k, c in zip(s["strike"], s["cp"])]
    return s, F, D, atm_iv, w


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="50ETF", choices=list(BOARD))
    ap.add_argument("--sigma", type=float, default=0.358, help="Stage 1 estimate for 600482")
    ap.add_argument("--r", type=float, default=0.015)
    ap.add_argument("--csv"); ap.add_argument("--expiry", help="YYYY-MM-DD, with --csv")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    check_greeks(r=a.r, sigma=a.sigma)

    # ---- Part C data
    today = dt.date.today()
    if a.csv:
        exps = [dt.date.fromisoformat(a.expiry)]
        chains = [pd.read_csv(a.csv)]
        label = a.csv
    elif a.demo:
        exps = next_expiries(today)
        chains = [demo_chain((e - today).days / 365, seed=i) for i, e in enumerate(exps)]
        label = "DEMO"
    else:
        exps, chains, label = [], [], a.underlying
        for e in next_expiries(today):
            try:
                c = fetch_chain(a.underlying, e)
                if len(c) >= 10:
                    exps.append(e); chains.append(c)
            except Exception as err:
                print(f"[fetch failed for {e}: {err}]")
        if not chains:
            raise SystemExit("No option data. Try --demo, or export a chain to CSV and use --csv/--expiry.")

    results = []
    print(f"\n[C] implied vols for {label}")
    for e, c in zip(exps, chains):
        T = (e - today).days / 365
        s, F, D, atm_iv, w = smile(c, T)
        results.append((e, T, s, F, D, atm_iv, w))
        lo, hi = s.iloc[0], s.iloc[-1]
        print(f"    expiry {e} (T = {T*365:.0f}d): forward {F:.4f}, implied rate {-np.log(D)/T:+.2%}, "
              f"ATM IV {atm_iv:.1%}")
        print(f"      IV at K/F={lo.moneyness:.2f}: {lo.iv:.1%}   at K/F={hi.moneyness:.2f}: {hi.iv:.1%}   "
              f"-> BSM says these should all equal {atm_iv:.1%}")
        worst = ((s.flat_price - s.price) / s.price).abs().max()
        print(f"      pricing every strike with the ATM vol misprices by up to {worst:.0%} of market price")

    # ---- plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    S = np.linspace(60, 140, 400)
    for T, lab in ((5 / 242, "1 week"), (21 / 242, "1 month"), (63 / 242, "3 months")):
        g = bsm(S, 100.0, T, a.r, a.sigma)
        ax[0, 0].plot(S, g["price"], label=lab)
        ax[0, 1].plot(S, g["delta"], label=lab)
        ax[0, 2].plot(S, g["gamma"], label=lab)
    ax[0, 0].plot(S, np.maximum(S - 100, 0), "k:", label="payoff at expiry")
    ax[0, 0].set_title(f"Call price ($\\sigma$ = {a.sigma:.1%}, K = 100)")
    ax[0, 1].set_title("Delta = shares needed to hedge")
    ax[0, 2].set_title("Gamma = how fast the hedge goes stale")
    for i in range(3):
        ax[0, i].set_xlabel("stock price (strike = 100)"); ax[0, i].legend(fontsize=8)

    for e, T, s, F, D, atm_iv, w in results:
        line, = ax[1, 0].plot(s.moneyness, s.iv, "o-", ms=4, label=f"{e} market IV")
        ax[1, 0].axhline(atm_iv, color=line.get_color(), ls="--", lw=1)
        ax[1, 1].plot(s.moneyness, (s.flat_price - s.price) / s.price, "o-", ms=4, label=str(e))
    ax[1, 0].set_title(f"{label} implied-vol smile (dashed = what BSM assumes)")
    ax[1, 0].set_xlabel("K / F"); ax[1, 0].set_ylabel("implied vol"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set_title("Pricing error of flat-vol BSM (model − market) / market")
    ax[1, 1].set_xlabel("K / F"); ax[1, 1].legend(fontsize=8)

    e, T, s, F, D, atm_iv, w = results[0]
    ax[1, 2].plot(w.index, w["C"] - w["P"], "o", label="market C − P")
    ax[1, 2].plot(w.index, D * (F - w.index), "r", label=f"D·(F − K), F = {F:.3f}")
    ax[1, 2].axhline(0, color="k", lw=0.5)
    ax[1, 2].set_title(f"Put–call parity → forward price ({e})")
    ax[1, 2].set_xlabel("strike"); ax[1, 2].legend(fontsize=8)

    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    fig.suptitle("Stage 2 — BSM Greeks (600482-like option) and the implied-vol smile", fontsize=13)
    fig.tight_layout()
    out = f"stage2_bsm_iv_{label.replace('.csv', '')}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
