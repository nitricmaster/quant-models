"""
Runs Stage 1-4 in one go and files the results by trading date.

  latest/                  newest figures (README can embed these, always current)
  reports/YYYY-MM-DD/      dated archive: figures + the printed numbers (log.txt)
  reports/history.csv      one row per trading day: sigma, mu, price, ATM implied vol ...

Holdings are read from environment variables so they never sit in the code:
  SHARES (default 1000)   CASH (default 100000)   RATE (default 0.011)

Usage
  python run_daily.py            # real data; does nothing if today's data is already filed
  python run_daily.py --force    # re-run anyway
  python run_daily.py --demo     # offline test
"""
import argparse, csv, os, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYMBOL = "600482"


def latest_quote(symbol):
    """Last trading date and unadjusted close (the price you see in a trading app)."""
    try:
        import akshare as ak
        fn = getattr(ak, "stock_zh_a_hist_em", None) or ak.stock_zh_a_hist
        df = fn(symbol=symbol, period="daily", start_date="20260101", adjust="")
        return str(df["日期"].iloc[-1])[:10], float(df["收盘"].iloc[-1])
    except Exception:
        import yfinance as yf
        df = yf.download(symbol + ".SS", period="1mo", auto_adjust=False, progress=False)
        return str(df.index[-1].date()), float(df["Close"].squeeze().iloc[-1])


def run(script, *args):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                       cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8")
    ok = p.returncode == 0
    print(f"  {script:26s} {'ok' if ok else 'FAILED'}")
    return ok, p.stdout + ("" if ok else "\n[stderr]\n" + p.stderr[-1500:])


def grab(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    shares, cash, rate = os.getenv("SHARES", "1000"), os.getenv("CASH", "100000"), os.getenv("RATE", "0.011")

    if a.demo:
        date, price, extra = "demo", 28.5, ["--demo"]
    else:
        date, price = latest_quote(SYMBOL)
        extra = []
    out = HERE / "reports" / date
    if out.exists() and not (a.force or a.demo):
        print(f"{date} already filed (market closed or already run). Use --force to redo.")
        return
    print(f"Trading date {date}, close {price}")

    jobs = [
        ("stage1_gbm.py", ["--symbol", SYMBOL]),
        ("stage2_bsm_iv.py", ["--r", rate]),
        ("stage3_delta_hedge.py", ["--symbol", SYMBOL, "--r", rate]),
        ("stage4_my_portfolio.py", ["--symbol", SYMBOL, "--shares", shares, "--cash", cash,
                                    "--r", rate, "--price", price]),
    ]
    logs, all_ok = [], True
    for script, args in jobs:
        ok, text = run(script, *args, *extra)
        all_ok &= ok
        logs.append(f"##### {script}\n{text}")
    log = "\n".join(logs)

    out.mkdir(parents=True, exist_ok=True)
    (HERE / "latest").mkdir(exist_ok=True)
    (out / "log.txt").write_text(log, encoding="utf-8")
    for png in HERE.glob("stage*_*.png"):
        stage = png.name.split("_")[0]                      # stage1.png, stage2.png ...
        shutil.copy(png, HERE / "latest" / f"{stage}.png")
        shutil.move(str(png), out / png.name)

    if not a.demo:                                          # one summary row per trading day
        row = {
            "date": date, "close": price,
            "sigma": grab(r"sigma \(annual\) =\s*([\d.]+%)", log),
            "mu": grab(r"mu    \(annual\) =\s*(-?[\d.]+%)", log),
            "excess_kurtosis": grab(r"excess kurtosis = (-?[\d.]+)", log),
            "etf_atm_iv": grab(r"ATM IV ([\d.]+%)", log),
            "hedge_pnl_mean": grab(r"P&L / premium: mean ([+-][\d.]+%)", log),
            "stock_weight": grab(r"stock weight ([\d.]+%)", log),
            "var_1d_hist": grab(r"historical ([\d,]+)", log),
        }
        hist = HERE / "reports" / "history.csv"
        new = not hist.exists()
        with hist.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if new:
                w.writeheader()
            w.writerow(row)
    print(f"Filed under reports/{date}/ and latest/" + ("" if all_ok else "  (some stages failed, see log.txt)"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
