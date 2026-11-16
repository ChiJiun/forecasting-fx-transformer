"""從學長 FX-predect-v2 專案的 Datastream 原始檔抽出月頻面板，寫到 data/monthly/。

輸出寬表（index = 參考月份 YYYY-MM，欄 = 11 個貨幣單位 + us）：
  fx.csv    匯率（外幣 / 1 美元），**該月月初（約第 2 天）的單點取值**
  cpi.csv   CPI 指數，參考月份 = 物價所屬月份（尚未處理發布延遲）
  rate.csv  3 個月期短率（年化 %），參考月份 = 該月的值（尚未處理發布延遲）
  money.csv 貨幣量（M1 或 M0，見 MONEY_PREFIX），僅 MACRO_SUBSET + us
  ip.csv    工業生產指數，僅 MACRO_SUBSET + us

發布延遲在 experiments/monthly/mfx.py 的 RELEASE_LAG 處理，這裡只對齊參考月份。

用法：python experiments/monthly/build_panel.py [--src D:/Code/FX-predect-v2-main/fx/data]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "monthly"
DEM_PER_EUR = 1.95583  # 1999-01-01 不可撤銷換算率

CURRENCIES = ["australia", "canada", "denmark", "eurozone", "japan", "south_korea",
              "norway", "sweden", "switzerland", "taiwan", "uk"]

FX_COLS = {
    "australia": "AUSTRALIAN $ TO US $ (RFV) - EXCHANGE RATE",
    "canada": "CANADIAN $ TO US $ (RFV) - EXCHANGE RATE",
    "denmark": "DANISH KRONE TO US $ (RFV) - EXCHANGE RATE",
    "eurozone": "EURO TO US $ (RFV) - EXCHANGE RATE",
    "japan": "JAPANESE YEN TO US $ (RFV) - EXCHANGE RATE",
    "south_korea": "SOUTH KOREAN WON TO US $ (RFV) - EXCHANGE RATE",
    "norway": "NORWEGIAN KRONE TO US $(RFV) - EXCHANGE RATE",
    "sweden": "SWEDISH KRONA TO US (RFV) - EXCHANGE RATE",
    "switzerland": "SWISS FRANC TO US $ (RFV) - EXCHANGE RATE",
    "taiwan": "TAIWAN NEW $ TO US $ (RFV) - EXCHANGE RATE",
    "uk": "UK POUND TO US $ (RFV) - EXCHANGE RATE",
}
DEM_COL = "GERMAN MARK TO US $ (WMR) - EXCHANGE RATE"

# 歐元區沒有總體 CPI 欄位，用德國 CPI 代理（限制寫在報告）
CPI_COLS = {
    "australia": "AU CPI, Standardized (based on source quarterly data), SA, Index, 2010=100",
    "canada": "CN CPI NADJ",
    "denmark": "DK CPI NADJ",
    "eurozone": "BD CPI: TOTAL NADJ",
    "japan": "JP CPI: NATIONAL MEASURE SADJ",
    "south_korea": "KO CPI NADJ",
    "norway": "NW CPI NADJ",
    "sweden": "SD CPI NADJ",
    "switzerland": "SW CPI (2020M12=100) NADJ",
    "taiwan": "TW CPI SADJ",
    "uk": "UK CPI INDEX 00 : ALL ITEMS- ESTIMATED PRE-1997 NADJ",
    "us": "US CPI - ALL URBAN: ALL ITEMS SADJ",
}

# 3 個月期短率；list 代表依序優先（前者有值用前者，缺值用後者補）
RATE_COLS = {
    "australia": ["AUSTRALIA DEALER BILL 90 D - MIDDLE RATE"],
    "canada": ["CAN 3M T-BILL (USE CNTBB3M) - MIDDLE RATE"],
    "denmark": ["DANISH KRONE 3M DEPOSIT (FT/RFV) - MIDDLE RATE"],
    "eurozone": ["EBF EURIBOR 3M DELAYED - OFFERED RATE"],
    "japan": ["JAPAN INTERBANK JPY 3M - OFFERED RATE"],
    "south_korea": [("bench", "KOREA NCD 91 DAYS - MIDDLE RATE")],
    "norway": ["NORWAY DEPOSIT 3 MONTH - MIDDLE RATE"],
    "sweden": ["SWEDEN IBK. STIBOR 3M DELAYED - MIDDLE RATE"],
    "switzerland": ["SWISS INTERBANK 3M (ZRC:SNB) 'DEAD' - BID RATE",
                    "SWISS FRANC 3M DEPOSIT (FT/RFV) - MIDDLE RATE"],
    "taiwan": ["TAIWAN MONEY MARKET 90 DAYS - MIDDLE RATE"],
    "uk": ["UK TREASURY BILL TENDER 3M - MIDDLE RATE",
           "UK INTERBANK 3 MONTH 'DEAD' - MIDDLE RATE"],
    "us": [("macro", "US TREASURY BILL RATE - 3 MONTH (EP) NADJ")],
}


# 貨幣模型／含產出缺口的 Taylor rule 只在貨幣量與 IP 水準值都可得的子集上做（MACRO_SUBSET）。
# 貨幣量依各國可得性：日、韓、台、美為 M1，英國、瑞典只有 M0（窄貨幣）。
# 美國 M1 在 2020-05 重新定義（log 跳 1.2），在 mfx.py 以接合處理；原始值在此不改。
MONEY_PREFIX = {
    "japan": "JP MONEY SUPPLY: M1",
    "south_korea": "KO MONEY SUPPLY - M1",
    "taiwan": "TW MONEY SUPPLY - M1A",
    "uk": "UK MONEY SUPPLY M0",
    "sweden": "SD MONEY SUPPLY - M0",
    "us": "US MONEY SUPPLY M1 CURN",
}
IP_PREFIX = {
    "japan": "JP INDUSTRIAL PRODUCTION - MINING & MANUFACTURING",
    "south_korea": "KO INDUSTRIAL PRODUCTION VOLA",
    "taiwan": "TW INDUSTRIAL PRODUCTION INDEX VOLA",
    "uk": "UK INDUSTRIAL PRODUCTION INDEX - MANUFACTURING",
    "sweden": "SD INDUSTRIAL PRODUCTION - MINING & MANUFACTURING",
    "us": "US INDUSTRIAL PRODUCTION - TOTAL INDEX",
}


def _by_prefix(df, prefix):
    hits = [c for c in df.columns if c.startswith(prefix) and "%YOY" not in c.upper()]
    if len(hits) != 1:
        raise ValueError(f"欄位前綴 {prefix!r} 對到 {hits}")
    return hits[0]


def _month_index(s):
    return pd.to_datetime(s.astype(str).str.replace("/", "-"), format="mixed").dt.to_period("M")


def _read(path, date_col):
    df = pd.read_csv(path)
    df.index = _month_index(df[date_col])
    df = df.drop(columns=date_col)
    if df.index.duplicated().any():
        raise ValueError(f"{path} 參考月份重複")
    return df


def build(src):
    src = Path(src)
    fx_raw = _read(src / "non_organized_data/fx_data/匯率.csv", "Name")
    feat = _read(src / "organized_data/raw_data/feature.csv", "Date")
    rate_raw = _read(src / "organized_data/raw_data/interest.csv", "Date")
    bench = _read(src / "non_organized_data/interest_rate/Benchmark 3 month.csv", "Name")
    us_macro = _read(src / "non_organized_data/monthly_data/US.csv", "Name")
    sources = {"bench": bench, "macro": us_macro}

    fx = pd.DataFrame({c: fx_raw[col] for c, col in FX_COLS.items()})
    synth = fx_raw[DEM_COL] / DEM_PER_EUR
    pre = fx["eurozone"].isna()
    fx.loc[pre, "eurozone"] = synth[pre]
    # 重疊期間檢查合成歐元與真實歐元一致
    both = fx_raw["EURO TO US $ (RFV) - EXCHANGE RATE"].notna() & synth.notna()
    if both.any():
        rel = (synth[both] / fx_raw.loc[both, "EURO TO US $ (RFV) - EXCHANGE RATE"] - 1).abs()
        print(f"合成歐元 vs 真實歐元：重疊 {both.sum()} 月，最大相對差 {rel.max():.4%}")

    cpi = pd.DataFrame({c: feat[col] for c, col in CPI_COLS.items()})
    money = pd.DataFrame({c: feat[_by_prefix(feat, p)] for c, p in MONEY_PREFIX.items()})
    ip = pd.DataFrame({c: feat[_by_prefix(feat, p)] for c, p in IP_PREFIX.items()})

    rate = {}
    for c, cols in RATE_COLS.items():
        s = None
        for col in cols:
            frame, name = (sources[col[0]], col[1]) if isinstance(col, tuple) else (rate_raw, col)
            v = frame[name]
            s = v if s is None else s.combine_first(v)
        rate[c] = s
    rate = pd.DataFrame(rate)

    idx = pd.period_range("1986-01", "2023-10", freq="M")
    out = {}
    for name, df in (("fx", fx), ("cpi", cpi), ("rate", rate), ("money", money), ("ip", ip)):
        df = df.reindex(idx).astype(float)
        df.index = df.index.astype(str)
        df.index.name = "month"
        out[name] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/Code/FX-predect-v2-main/fx/data")
    args = ap.parse_args()
    out = build(args.src)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(OUT / f"{name}.csv", float_format="%.6g")
        cov = {c: f"{df[c].first_valid_index()}~{df[c].last_valid_index()}" for c in df.columns}
        print(f"[{name}]", cov)
    print("最後一列匯率（檢查報價方向：外幣/美元）:")
    print(out["fx"].dropna(how="all").iloc[-1].round(4).to_dict())


if __name__ == "__main__":
    main()
