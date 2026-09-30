"""데이터 불러오기: 파일(csv, xlsx, zip, json) 또는 링크. 형식을 자동으로 판별하고 표준 열 이름으로 맞춘다.

지원 형식
  eci      Epoch Capabilities Index (eci_scores.csv 또는 benchmark_data.zip)
  notable  Epoch Notable AI Models (notable_ai_models.csv)
  aa_pairs AA 지수와 ECI 대응표 (model, aa, eci)
  korea    한국 모델 AA 점수표 (model, aa, finalist)
  stock    컴퓨팅 보유 요약(json)
"""
import io, json, zipfile, hashlib
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "data" / "baseline"

KIND_LABEL = {"eci": "Epoch 역량지수(ECI)", "notable": "Epoch 주목 모델", "aa_pairs": "AA–ECI 대응표", "korea": "한국 모델 AA 점수", "stock": "컴퓨팅 보유 요약"}

DEFAULT_URLS = {
    "eci": "https://epoch.ai/data/benchmark_data.zip",
    "notable": "https://epoch.ai/data/notable_ai_models.csv",
}


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _norm_cols(df):
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def detect_kind(df: pd.DataFrame):
    cols = {c.lower() for c in df.columns}
    if "eci" in cols and ("accessibility group" in cols or "model accessibility" in cols) and "date" in cols:
        return "eci"
    if "training compute (flop)" in cols and "publication date" in cols:
        return "notable"
    if {"aa", "eci"} <= cols:
        return "aa_pairs"
    if "aa" in cols and "model" in cols:
        return "korea"
    return None


def normalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = _norm_cols(df.copy())
    low = {c.lower(): c for c in df.columns}
    if kind == "eci":
        if "Accessibility group" not in df.columns and "model accessibility" in low:
            acc = df[low["model accessibility"]].astype(str).str.lower()
            df["Accessibility group"] = acc.map(lambda s: "Open weights" if "open" in s else "Closed weights")
        df["date"] = pd.to_datetime(df[low["date"]], errors="coerce")
        df["eci"] = pd.to_numeric(df[low["eci"]], errors="coerce")
        if "Country (of organization)" not in df.columns:
            df["Country (of organization)"] = ""
        if "Model" not in df.columns and "model" in low:
            df["Model"] = df[low["model"]]
        df = df.dropna(subset=["date", "eci"])
    elif kind == "notable":
        df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
        df["flop"] = pd.to_numeric(df["Training compute (FLOP)"], errors="coerce")
    elif kind == "aa_pairs":
        df = df.rename(columns={low["aa"]: "aa", low["eci"]: "eci"})
        if "model" not in df.columns and "model" in low: df = df.rename(columns={low["model"]: "model"})
        df["aa"] = pd.to_numeric(df["aa"], errors="coerce"); df["eci"] = pd.to_numeric(df["eci"], errors="coerce")
        df = df.dropna(subset=["aa", "eci"])
    elif kind == "korea":
        df = df.rename(columns={low["aa"]: "aa", low["model"]: "model"})
        df["aa"] = pd.to_numeric(df["aa"], errors="coerce")
        if "aa_low" not in df.columns: df["aa_low"] = df["aa"]
        if "finalist" not in df.columns: df["finalist"] = False
        df["finalist"] = df["finalist"].astype(str).str.lower().isin(["true", "1", "yes", "y", "예", "o"])
    return df


def read_bytes(name: str, b: bytes):
    """파일 내용을 읽어 [(kind, df 또는 dict, 설명)] 목록을 돌려준다."""
    out = []
    n = name.lower()
    if n.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(b))
        for zi in z.namelist():
            if zi.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(io.BytesIO(z.read(zi)), low_memory=False, encoding="utf-8-sig")
                except Exception:
                    continue
                k = detect_kind(df)
                if k == "eci" and "eci_scores" not in zi.lower() and any("eci_scores" in x.lower() for x in z.namelist()):
                    continue
                if k: out.append((k, normalize(df, k), f"{name} 안의 {zi}"))
        return out
    if n.endswith(".json"):
        d = json.loads(b.decode("utf-8-sig"))
        if "KR_position_2025" in d or "global_stock_H100e_yearend" in d:
            out.append(("stock", d, name))
        return out
    if n.endswith(".xlsx") or n.endswith(".xls"):
        sheets = pd.read_excel(io.BytesIO(b), sheet_name=None)
        for sh, df in sheets.items():
            k = detect_kind(_norm_cols(df))
            if k: out.append((k, normalize(df, k), f"{name}의 {sh} 시트"))
        return out
    df = pd.read_csv(io.BytesIO(b), low_memory=False, encoding="utf-8-sig")
    k = detect_kind(_norm_cols(df))
    if k: out.append((k, normalize(df, k), name))
    return out


def fetch_url(url: str, timeout=60):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "kcci-sovereign-ai-simulator"})
    r.raise_for_status()
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.csv"
    if "." not in name:
        ct = r.headers.get("content-type", "")
        name += ".zip" if "zip" in ct else ".csv"
    return name, r.content


def load_baseline():
    data = {}
    meta = {}
    for kind, fn in (("eci", "eci_scores.csv"), ("notable", "notable_ai_models.csv"), ("aa_pairs", "aa_eci_pairs.csv"), ("korea", "korea_models.csv")):
        b = (BASE / fn).read_bytes()
        df = pd.read_csv(io.BytesIO(b), low_memory=False, encoding="utf-8-sig")
        data[kind] = normalize(df, kind)
        meta[kind] = dict(source=f"기준본 {fn}", sha16=sha16(b), rows=len(df))
    b = (BASE / "compute_stock.json").read_bytes()
    data["stock"] = json.loads(b.decode("utf-8"))
    meta["stock"] = dict(source="기준본 compute_stock.json", sha16=sha16(b), rows=None)
    return data, meta


def diff_eci(old: pd.DataFrame, new: pd.DataFrame):
    """새 ECI 자료와 기준본 비교: 신규 모델, 점수 변경, 최고점 변화."""
    o = old.set_index("Model")["eci"]; n = new.set_index("Model")["eci"]
    added = sorted(set(n.index) - set(o.index)); removed = sorted(set(o.index) - set(n.index))
    common = sorted(set(n.index) & set(o.index))
    changed = [(m, float(o[m]) if not hasattr(o[m], "__len__") else float(o[m].iloc[0]), float(n[m]) if not hasattr(n[m], "__len__") else float(n[m].iloc[0])) for m in common]
    changed = [c for c in changed if abs(c[1] - c[2]) > 1e-6]
    def top(df, grp):
        s = df[df["Accessibility group"] == grp]
        return (float(s["eci"].max()), s.loc[s["eci"].idxmax(), "Model"]) if len(s) else (None, None)
    return dict(added=added, removed=removed, changed=changed,
                frontier_old=top(old, "Closed weights"), frontier_new=top(new, "Closed weights"),
                open_old=top(old, "Open weights"), open_new=top(new, "Open weights"),
                last_date_old=str(old["date"].max().date()), last_date_new=str(new["date"].max().date()))
