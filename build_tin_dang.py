"""Đóng gói kho tin đăng cho trang tìm nhà của người mua.

Giữ những cột mà người mua thật sự đọc khi lướt tin, kèm bốn dấu hiệu bất
thường đã chấm sẵn để mỗi tin có nhãn ngay trong danh sách kết quả mà không
phải chạy lại mô hình trên từng tin.

Bốn dấu hiệu được dựng lại từ chính giá mô hình đã ước tính ở notebook 03,
không đi qua biểu mẫu, nên điểm và kết luận trùng khít với bảng đã chấm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path("/home/dang.lu/Personal/Final ML")
APP = GOC / "GUI_Project1"
sys.path.insert(0, str(APP))

from scoring import load_artifacts, price_segment  # noqa: E402

df = pd.read_parquet(GOC / "Topic_1" / "data" / "nhatot_clean.parquet")
cu = pd.read_parquet(APP / "data" / "anomaly_scores.parquet")
art = load_artifacts()
meta, bands = art["meta"], art["bands"]
print(f"Đọc {len(df):,} tin đăng")

gia = df["gia_ban_ty"].to_numpy(float)
don_gia = df["don_gia_trieu_m2"].to_numpy(float)
pred_ty = cu["pred_ty"].to_numpy(float)

# ───────────────────────── 1. Chênh lệch so với giá mô hình ước tính
residual_log = np.log1p(gia) - np.log1p(pred_ty)
seg = np.array([price_segment(g) for g in gia])
med = np.array([meta["seg_median"].get(s, 0.0) for s in seg])
sig = np.array([meta["seg_sigma"].get(s) or 1e-6 for s in seg])
s_resid = np.clip(np.abs((residual_log - med) / sig), 0, 3) / 3.0

# ───────────────────────── 2 và 3. Vị trí trong dải giá của phường
b = bands.set_index("area_key")
lo = df["area_key"].map(b["lo"]).to_numpy(float)
hi = df["area_key"].map(b["hi"]).to_numpy(float)
p10 = df["area_key"].map(b["p10"]).to_numpy(float)
p90 = df["area_key"].map(b["p90"]).to_numpy(float)

s_min_max = ((don_gia < lo) | (don_gia > hi)).astype(float)
ngoai = (don_gia < p10) | (don_gia > p90)
d = np.where(ngoai, np.minimum(np.abs(don_gia - p10), np.abs(don_gia - p90)), 0.0)
s_pct = np.clip(d / (p90 - p10), 0, 1)

# ───────────────────────── 4. Isolation Forest
ifX = df[meta["if_features"]].fillna(pd.Series(meta["if_medians"]))
diem_if = art["iso"].decision_function(ifX)
s_ml = np.clip((meta["if_score_max"] - diem_if)
               / (meta["if_score_max"] - meta["if_score_min"]), 0, 1)

w = meta["weights"]
tong = 100.0 * (w["S_resid"] * s_resid + w["S_min_max"] * s_min_max
                + w["S_percentile"] * s_pct + w["S_ml"] * s_ml)
ket_luan = np.where(tong < meta["score_threshold"], "Bình thường",
                    np.where(residual_log > 0, "Giá quá đắt", "Giá quá rẻ"))
n_dau_hieu = ((s_resid > 0.50).astype("int8") + (s_min_max > 0).astype("int8")
              + (s_pct > 0.10).astype("int8") + (s_ml > 0.60).astype("int8"))

assert (ket_luan == cu["verdict_B"].to_numpy()).all(), "kết luận lệch bảng gốc"
assert (n_dau_hieu == cu["n_signals_B"].to_numpy()).all(), "số dấu hiệu lệch"
assert np.abs(tong - cu["score_B"].to_numpy()).max() < 1e-4, "điểm lệch"
print("Dựng lại bốn dấu hiệu: trùng khít bảng đã chấm ở notebook")

# ───────────────────────── 5. Mã tin, giữ nguyên mã của hàng đợi duyệt
ma = pd.Series(index=df.index, dtype=object)
co_co = cu[cu["verdict_B"] != "Bình thường"].sort_values(
    "score_B", ascending=False).index
ma.loc[co_co] = [f"NT{i:04d}" for i in range(1, len(co_co) + 1)]
con_lai = ma[ma.isna()].index
ma.loc[con_lai] = [f"NT{i:04d}" for i in range(len(co_co) + 1, len(df) + 1)]

# ───────────────────────── 6. Bảng đóng gói
ra = pd.DataFrame({
    "ma_tin": ma.to_numpy(),
    "tieu_de": df["tieu_de"].to_numpy(),
    "district": df["district"].astype("category"),
    "area_key": df["area_key"].astype("category"),
    "street": df["street"].astype("category"),
    "loai_hinh": df["loai_hinh"].astype("category"),
    "giay_to_phap_ly": df["giay_to_phap_ly"].astype("category"),
    "tinh_trang_noi_that": df["tinh_trang_noi_that"].astype("category"),
    "dac_diem": df["dac_diem"].astype("category"),
    "gia_ban_ty": gia.astype("float32"),
    "don_gia_trieu_m2": don_gia.astype("float32"),
    "dien_tich_m2": df["dien_tich_m2"].to_numpy("float32"),
    "so_phong_ngu": df["so_phong_ngu"].to_numpy("float32"),
    "so_phong_ve_sinh": df["so_phong_ve_sinh"].to_numpy("float32"),
    "tong_so_tang": df["tong_so_tang"].to_numpy("float32"),
    "chieu_ngang_m": df["chieu_ngang_m"].to_numpy("float32"),
    "pred_ty": pred_ty.astype("float32"),
    "score": tong.astype("float32"),
    "verdict": pd.Categorical(ket_luan),
    "n_signals": n_dau_hieu,
    "s_resid": s_resid.astype("float32"),
    "s_min_max": s_min_max.astype("float32"),
    "s_pct": s_pct.astype("float32"),
    "s_ml": s_ml.astype("float32"),
})

dich = APP / "data" / "tin_dang.parquet"
ra.to_parquet(dich, index=False, compression="zstd")
print(f"\nGhi {dich.name}: {ra.shape[0]:,} x {ra.shape[1]} "
      f"({dich.stat().st_size / 1024:.0f} KB)")
print(ra["verdict"].value_counts().to_dict())
