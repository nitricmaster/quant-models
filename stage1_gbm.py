"""
Stage 1 — Modelling a stock with Geometric Brownian Motion (GBM)
    dS_t = mu * S_t dt + sigma * S_t dW_t

What it does
  1. Loads daily prices (akshare -> yfinance fallback, or --demo synthetic data)
  2. Estimates mu and sigma from log returns (with a standard error for mu)
  3. Simulates GBM with Euler-Maruyama and with the exact Ito solution,
     driven by the SAME Brownian increments, and measures the strong error
  4. Tests the GBM assumptions against the data (fat tails, vol clustering)

Usage
  pip install akshare yfinance numpy pandas scipy matplotlib
  python stage1_gbm.py                  # 600482, last 5 years
  python stage1_gbm.py --symbol 600482 --start 20200101
  python stage1_gbm.py --demo           # no network needed
"""
import argparse
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

TRADING_DAYS = 242  # A-share trading days per year (US: 252)


# ---------------------------------------------------------------- data
def load_prices(symbol: str, start: str, demo: bool) -> pd.Series:
    """Daily adjusted closes. Fetched once per day and cached, so Stages 1, 3 and 4
    share one download instead of hitting the data source three times in a row."""
    if demo:
        return synthetic_prices()
    cache = Path(__file__).resolve().parent / ".cache" / f"{symbol}_{start}_{date.today():%Y%m%d}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        source = s.name                         # the cache's column header holds the source
    else:
        s, source = _download(symbol, start)
        cache.parent.mkdir(exist_ok=True)
        s.rename(source).to_csv(cache)
    print(f"[data source: {source}, {len(s)} rows, last {s.index[-1].date()}]")
    return s.rename(symbol)


def _download(symbol: str, start: str):
    try:
        import akshare as ak
        # adjust="hfq": back-adjusted for dividends/splits, so returns are correct
        fn = getattr(ak, "stock_zh_a_hist_em", None) or ak.stock_zh_a_hist
        df = fn(symbol=symbol, period="daily", start_date=start, adjust="hfq")
        s = pd.Series(df["收盘"].values, index=pd.to_datetime(df["日期"]), name=symbol)
        if len(s) > 50:
            return s.astype(float), "akshare/eastmoney"
    except Exception as e:
        print(f"[akshare failed: {str(e)[:80]}] trying yfinance ...")
    import yfinance as yf
    suffix = ".SS" if symbol.startswith(("6", "9")) else ".SZ"
    start_iso = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    df = yf.download(symbol + suffix, start=start_iso, auto_adjust=True, progress=False)
    s = df["Close"].squeeze().dropna().rename(symbol)
    if len(s) < 50:
        raise RuntimeError("both akshare and yfinance returned no data")
    return s, "yfinance"


def synthetic_prices(n=1200, seed=7) -> pd.Series:
    """Fat-tailed, vol-clustered fake data (GARCH(1,1) + Student-t) for --demo."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    var, r = omega / (1 - alpha - beta), np.empty(n)
    for i in range(n):
        z = rng.standard_t(5) / np.sqrt(5 / 3)
        r[i] = 3e-4 + np.sqrt(var) * z
        var = omega + alpha * (r[i] - 3e-4) ** 2 + beta * var
    idx = pd.bdate_range(end="2026-11-13", periods=n)
    return pd.Series(20 * np.exp(np.cumsum(r)), index=idx, name="DEMO")


# ---------------------------------------------------------- estimation
def estimate_gbm(prices: pd.Series):
    """Under GBM, log returns are iid N((mu - sigma^2/2) dt, sigma^2 dt)."""
    r = np.log(prices).diff().dropna()
    dt = 1 / TRADING_DAYS
    T = len(r) * dt                               # sample length in years
    sigma = r.std(ddof=1) / np.sqrt(dt)
    mu = r.mean() / dt + 0.5 * sigma**2           # Ito correction
    se_mu = sigma / np.sqrt(T)                    # depends on T only, not on sampling frequency
    se_sigma = sigma / np.sqrt(2 * len(r))
    return r, mu, sigma, se_mu, se_sigma


# ---------------------------------------------------------- simulation
def simulate(S0, mu, sigma, T, n_steps, n_paths, seed=0):
    """Euler-Maruyama vs exact solution on the same Brownian paths."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    t = np.linspace(dt, T, n_steps)
    exact = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    em = S0 * np.cumprod(1 + mu * dt + sigma * dW, axis=1)
    pad = np.full((n_paths, 1), S0)
    return np.hstack([pad, exact]), np.hstack([pad, em])


def strong_error(S0, mu, sigma, T=1.0, n_paths=20000):
    """E|S_T^EM - S_T^exact| for several step sizes; theory: O(dt^0.5)."""
    steps = np.array([8, 16, 32, 64, 128, 256])
    errs = []
    for n in steps:
        ex, em = simulate(S0, mu, sigma, T, n, n_paths, seed=1)
        errs.append(np.abs(ex[:, -1] - em[:, -1]).mean())
    slope = np.polyfit(np.log(T / steps), np.log(errs), 1)[0]
    return T / steps, np.array(errs), slope


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600482")
    ap.add_argument("--start", default="20211101")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    prices = load_prices(a.symbol, a.start, a.demo)
    r, mu, sigma, se_mu, se_sigma = estimate_gbm(prices)
    S0 = float(prices.iloc[-1])
    jb = stats.jarque_bera(r)
    acf_sq = pd.Series(r.values**2).autocorr(1)

    print(f"\n=== {prices.name}: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(r)} returns) ===")
    print(f"sigma (annual) = {sigma:6.2%}  ± {se_sigma:.2%}")
    print(f"mu    (annual) = {mu:6.2%}  ± {se_mu:.2%}   <- compare the two standard errors")
    print(f"Ito correction sigma^2/2 = {0.5*sigma**2:.2%} per year (mean vs median growth)")
    print(f"skew = {stats.skew(r):.2f}, excess kurtosis = {stats.kurtosis(r):.2f} (GBM: 0, 0)")
    print(f"Jarque-Bera p-value = {jb.pvalue:.2e}  (normality {'rejected' if jb.pvalue < .01 else 'not rejected'})")
    print(f"lag-1 autocorr of squared returns = {acf_sq:.2f} (GBM: 0 -> vol clustering if > 0)")
    print(f"days with |return| > 9.5% (price-limit hits): {(r.abs() > 0.095).sum()}")

    # ---- plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    name = prices.name

    ax[0, 0].plot(prices.index, prices.values, lw=1)
    ax[0, 0].set_title(f"{name} adjusted close")

    T, n = 1.0, TRADING_DAYS
    ex, em = simulate(S0, mu, sigma, T, n, 2000)
    t = np.linspace(0, T, n + 1)
    ax[0, 1].plot(t, ex[:40].T, lw=0.6, alpha=0.5, color="tab:blue")
    q05, q50, q95 = np.percentile(ex, [5, 50, 95], axis=0)
    ax[0, 1].plot(t, q50, "k", lw=2, label="median  $S_0e^{(\\mu-\\sigma^2/2)t}$")
    ax[0, 1].plot(t, ex.mean(0), "r--", lw=2, label="mean  $S_0e^{\\mu t}$")
    ax[0, 1].fill_between(t, q05, q95, color="gray", alpha=0.2, label="5–95% band")
    ax[0, 1].set_title("1-year GBM fan (mean > median: Itô term)")
    ax[0, 1].set_xlabel("years"); ax[0, 1].legend(fontsize=8)

    ex1, em1 = simulate(S0, mu, sigma, T, 12, 1, seed=3)   # coarse grid so the gap is visible
    tc = np.linspace(0, T, 13)
    ax[0, 2].plot(tc, ex1[0], "o-", label="exact solution")
    ax[0, 2].plot(tc, em1[0], "s--", label="Euler–Maruyama (monthly steps)")
    ax[0, 2].set_title("Same Brownian path: EM vs exact"); ax[0, 2].legend(fontsize=8)

    dts, errs, slope = strong_error(S0, mu, sigma)
    ax[1, 0].loglog(dts, errs, "o-", label=f"measured, slope = {slope:.2f}")
    ax[1, 0].loglog(dts, errs[0] * (dts / dts[0]) ** 0.5, "k:", label="reference slope 0.5")
    ax[1, 0].set_title("EM strong convergence"); ax[1, 0].set_xlabel("dt")
    ax[1, 0].set_ylabel("$E|S_T^{EM}-S_T|$"); ax[1, 0].legend(fontsize=8)

    x = np.linspace(r.min(), r.max(), 400)
    ax[1, 1].hist(r, bins=80, density=True, alpha=0.6, label="actual log returns")
    ax[1, 1].plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r", label="normal (GBM)")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylim(bottom=0.05)
    ax[1, 1].set_title(f"Fat tails (excess kurtosis {stats.kurtosis(r):.1f})"); ax[1, 1].legend(fontsize=8)

    roll = r.rolling(30).std() * np.sqrt(TRADING_DAYS)
    ax[1, 2].plot(roll.index, roll.values, lw=1, label="30-day realised vol")
    ax[1, 2].axhline(sigma, color="r", ls="--", label=f"constant $\\sigma$ = {sigma:.0%}")
    ax[1, 2].set_title("Volatility is not constant"); ax[1, 2].legend(fontsize=8)

    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    ax[0, 0].tick_params(axis="x", rotation=30)
    ax[1, 2].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Stage 1 — GBM model of {name}:  $\\hat\\mu$ = {mu:.1%} ± {se_mu:.1%},  "
                 f"$\\hat\\sigma$ = {sigma:.1%} ± {se_sigma:.1%}", fontsize=13)
    fig.tight_layout()
    out = f"stage1_gbm_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
