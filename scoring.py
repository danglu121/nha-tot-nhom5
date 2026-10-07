"""Lớp suy luận dùng chung cho ứng dụng web.

Nhận thông tin một tin đăng, dựng đủ 49 tính năng mà mô hình cần, rồi trả về
giá gợi ý và điểm bất thường.

Mọi thang chuẩn hoá đều đọc từ models/metadata.json, chính là thang đã dùng
khi chấm điểm cả lô 7 935 tin ở notebook 03. Nếu tính lại thang tại thời điểm
chạy, điểm của một tin mới sẽ không so sánh được với lịch sử.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).parent
MODELS = APP_DIR / "models"

TEXT_FLAGS = {
    "txt_hem_xe_hoi":   r"hem xe hoi|hem oto|hem o to|hem xe hop",
    "txt_mat_tien":     r"mat tien|mat pho",
    "txt_no_hau":       r"no hau",
    "txt_nha_nat":      r"nha nat|nha cu nat|xuong cap",
    "txt_hoan_cong":    r"hoan cong",
    "txt_so_hong":      r"so hong|so do",
    "txt_thang_may":    r"thang may",
    "txt_dong_tien":    r"dong tien|cho thue|thu nhap",
    "txt_kinh_doanh":   r"kinh doanh|kd",
    "txt_gan_tien_ich": r"gan cho|gan truong|gan benh vien|gan sieu thi|gan metro",
    "txt_can_goc":      r"can goc|lo goc|hai mat tien|2 mat tien",
    "txt_thuong_luong": r"thuong luong|giam gia|ban gap|can ban gap",
    "txt_quy_hoach":    r"quy hoach|lo gioi|tranh chap",
}
_TRET_LAU_RE = re.compile(r"(\d+)\s*(?:tret|ham|lung|lau|tang)")
SUSPICIOUS_FLOORS = {11, 22, 23, 33, 44}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def floors_from_text(text: str) -> float:
    if not text:
        return np.nan
    found = [int(n) for n in _TRET_LAU_RE.findall(text) if n.isdigit()]
    found = [n for n in found if 1 <= n <= 12]
    return float(sum(found)) if found else np.nan


@lru_cache(maxsize=1)
def load_artifacts():
    """Load once and cache. Streamlit reruns the script on every interaction."""
    meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    return {
        "price_model": joblib.load(MODELS / "price_model.joblib"),
        "iso": joblib.load(MODELS / "isolation_forest.joblib"),
        "bands": pd.read_parquet(MODELS / "area_bands.parquet"),
        "meta": meta,
    }


def build_feature_row(listing: dict, art: dict) -> pd.DataFrame:
    """Turn a form submission into the exact 49-column row the model expects."""
    meta = art["meta"]
    bands = art["bands"]
    area_key = listing["area_key"]
    band = bands[bands["area_key"] == area_key]
    band = band.iloc[0] if len(band) else None

    title = listing.get("tieu_de", "") or ""
    desc = listing.get("mo_ta", "") or ""
    blob = strip_accents(f"{title} {desc}").lower()

    area = float(listing["dien_tich_m2"])
    used_area = listing.get("dien_tich_su_dung_m2")
    width = listing.get("chieu_ngang_m")
    depth = listing.get("chieu_dai_m")
    bedrooms = listing.get("so_phong_ngu")
    floors = listing.get("tong_so_tang")

    rect_area = (width * depth) if (width and depth) else np.nan
    txt_floors = floors_from_text(blob)

    row = {
        # --- numeric --------------------------------------------------
        "dien_tich_m2": area,
        "dien_tich_su_dung_m2": used_area if used_area else np.nan,
        "chieu_ngang_m": width if width else np.nan,
        "chieu_dai_m": depth if depth else np.nan,
        "so_phong_ngu": float(bedrooms) if bedrooms else np.nan,
        "so_phong_ve_sinh": float(listing["so_phong_ve_sinh"]) if listing.get("so_phong_ve_sinh") else np.nan,
        "tong_so_tang": float(floors) if floors else np.nan,
        "floor_area_ratio": (used_area / area) if (used_area and area) else np.nan,
        "area_per_bedroom": (area / bedrooms) if bedrooms else np.nan,
        "aspect_ratio": (depth / width) if (width and depth) else np.nan,
        "rect_area_m2": rect_area,
        "rect_area_gap_pct": (abs(rect_area - area) / area * 100) if rect_area == rect_area else np.nan,
        "price_hist_last": float(band["price_hist_last"]) if band is not None else np.nan,
        "price_hist_mean": float(band["price_hist_mean"]) if band is not None else np.nan,
        "price_hist_trend": float(band["price_hist_trend"]) if band is not None else np.nan,
        "price_hist_len": 13.0,
        "title_len": float(len(title)),
        "desc_len": float(len(desc)),
        "n_flags": 0.0,
        "tong_so_tang_filled": float(floors) if floors else txt_floors,
        "txt_floors": txt_floors,
        # --- categorical ----------------------------------------------
        "loai_hinh": listing.get("loai_hinh") or "(thiếu)",
        "giay_to_phap_ly": listing.get("giay_to_phap_ly") or "(thiếu)",
        "tinh_trang_noi_that": listing.get("tinh_trang_noi_that") or "(thiếu)",
        "huong_cua_chinh": listing.get("huong_cua_chinh") or "(thiếu)",
        "dac_diem": listing.get("dac_diem") or "(thiếu)",
        "district": (band["district"] if band is not None else area_key.split(" / ")[0]),
        "area_key": area_key,
        "street": listing.get("street") or "(thiếu)",
        # --- boolean ---------------------------------------------------
        "has_phone": 1,
        "so_phong_ngu_capped": 0,
        "so_phong_ve_sinh_capped": 0,
        "flag_floors_suspicious": int(floors in SUSPICIOUS_FLOORS) if floors else 0,
        "flag_rect_mismatch": 0,
        "flag_no_history": int(band is None),
        "flag_floors_from_text": int(not floors and txt_floors == txt_floors),
    }
    for name, pattern in TEXT_FLAGS.items():
        row[name] = int(bool(re.search(pattern, blob)))

    if row["rect_area_gap_pct"] == row["rect_area_gap_pct"]:
        row["flag_rect_mismatch"] = int(row["rect_area_gap_pct"] > 25)
    row["n_flags"] = float(
        row["flag_floors_suspicious"] + row["flag_rect_mismatch"] + row["flag_no_history"]
    )

    frame = pd.DataFrame([row])
    # Reindex to the exact training column order; a mismatch here silently
    # feeds the model the wrong values.
    return frame.reindex(columns=meta["feature_cols"])


def predict_price(listing: dict) -> dict:
    art = load_artifacts()
    X = build_feature_row(listing, art)
    pred_log = float(art["price_model"].predict(X)[0])
    pred_log = min(pred_log, art["meta"]["cap_log"])
    pred_ty = float(np.expm1(pred_log))

    # Confidence band from the measured MAPE of the matching price segment.
    mape = segment_mape(pred_ty)
    return {
        "pred_ty": pred_ty,
        "pred_unit_price": pred_ty * 1000 / listing["dien_tich_m2"],
        "low_ty": pred_ty * (1 - mape / 100),
        "high_ty": pred_ty * (1 + mape / 100),
        "segment_mape": mape,
        "X": X,
    }


# MAPE per price segment, measured on the held-out test set in notebook 02.
SEGMENT_MAPE = {"<3": 25.66, "3-5": 13.92, "5-7": 11.05, "7-10": 12.30,
                "10-15": 18.54, "15-25": 19.06, ">25": 26.65}


def price_segment(price_ty: float) -> str:
    bins = [0, 3, 5, 7, 10, 15, 25, np.inf]
    labels = ["<3", "3-5", "5-7", "7-10", "10-15", "15-25", ">25"]
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        if lo < price_ty <= hi:
            return label
    return ">25"


def segment_mape(price_ty: float) -> float:
    return SEGMENT_MAPE[price_segment(price_ty)]


def score_anomaly(listing: dict, asking_price_ty: float) -> dict:
    """Four signals and the composite score for one listing."""
    art = load_artifacts()
    meta, bands = art["meta"], art["bands"]
    area = float(listing["dien_tich_m2"])
    unit_price = asking_price_ty * 1000 / area

    prediction = predict_price(listing)
    X = prediction["X"]

    # --- Signal 1: Residual-Z, standardised within the price segment -----
    residual_log = float(np.log1p(asking_price_ty) - np.log1p(prediction["pred_ty"]))
    seg = price_segment(asking_price_ty)
    med = meta["seg_median"].get(seg, 0.0)
    sigma = meta["seg_sigma"].get(seg) or 1e-6
    z = (residual_log - med) / sigma
    s_resid = float(np.clip(abs(z), 0, 3) / 3.0)

    # --- Signal 2: Min/Max violation --------------------------------------
    band = bands[bands["area_key"] == listing["area_key"]]
    band = band.iloc[0] if len(band) else None
    if band is not None:
        lo, hi = float(band["lo"]), float(band["hi"])
        p10, p90 = float(band["p10"]), float(band["p90"])
    else:
        lo, hi = meta["expert_floor"], meta["expert_ceil"]
        p10, p90 = np.nan, np.nan
    s_min_max = float(unit_price < lo or unit_price > hi)

    # --- Signal 3: distance outside P10-P90, in band widths ---------------
    if np.isfinite(p10) and np.isfinite(p90) and p90 > p10:
        d = 0.0
        if unit_price < p10 or unit_price > p90:
            d = min(abs(unit_price - p10), abs(unit_price - p90))
        s_pct = float(np.clip(d / (p90 - p10), 0, 1))
    else:
        d, s_pct = 0.0, 0.0

    # --- Signal 4: Isolation Forest ---------------------------------------
    if_row = {f: X[f].iloc[0] if f in X.columns else np.nan for f in meta["if_features"]}
    if_row["don_gia_trieu_m2"] = unit_price
    if_frame = pd.DataFrame([if_row])[meta["if_features"]]
    if_frame = if_frame.fillna(pd.Series(meta["if_medians"]))
    if_score = float(art["iso"].decision_function(if_frame)[0])
    lo_s, hi_s = meta["if_score_min"], meta["if_score_max"]
    s_ml = float(np.clip((hi_s - if_score) / (hi_s - lo_s), 0, 1))

    # --- Composite ---------------------------------------------------------
    w = meta["weights"]
    total = 100.0 * (w["S_resid"] * s_resid + w["S_min_max"] * s_min_max
                     + w["S_percentile"] * s_pct + w["S_ml"] * s_ml)

    flagged = total >= meta["score_threshold"]
    verdict = ("Bình thường" if not flagged
               else ("Giá quá đắt" if residual_log > 0 else "Giá quá rẻ"))

    return {
        "verdict": verdict,
        "total_score": total,
        "threshold": meta["score_threshold"],
        "unit_price": unit_price,
        "pred_ty": prediction["pred_ty"],
        "residual_log": residual_log,
        "z": z,
        "signals": {
            "Residual-Z": s_resid,
            "Min/Max": s_min_max,
            "P10–P90": s_pct,
            "Isolation Forest": s_ml,
        },
        "band": {"lo": lo, "hi": hi, "p10": p10, "p90": p90,
                 "fallback": bool(band["fallback"]) if band is not None else True},
        "n_signals": int((s_resid > 0.50) + (s_min_max > 0) + (s_pct > 0.10) + (s_ml > 0.60)),
    }
