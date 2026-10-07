"""Bước 6 — ứng dụng web Nhà Tốt: gợi ý giá và phát hiện bất thường.

Chạy:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from scoring import load_artifacts, predict_price, score_anomaly

# Thư mục triển khai tự chứa: mô hình và dữ liệu chấm điểm nằm cạnh app.py.
ROOT = Path(__file__).parent

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
VERDICT_STYLE = {
    "Bình thường": ("#1baf7a", "✓"),
    "Giá quá rẻ": ("#2a78d6", "↓"),
    "Giá quá đắt": ("#eb6834", "↑"),
}

st.set_page_config(page_title="Nhà Tốt — Định giá & Cảnh báo giá",
                   page_icon="🏠", layout="wide")


@st.cache_resource
def artifacts():
    return load_artifacts()


@st.cache_data
def load_scores():
    return pd.read_parquet(ROOT / "data" / "anomaly_scores.parquet")


art = artifacts()
meta = art["meta"]
bands = art["bands"]

st.sidebar.title("🏠 Nhà Tốt")
st.sidebar.caption("Định giá và cảnh báo giá bất thường")
page = st.sidebar.radio(
    "Chức năng",
    ["Định giá nhà", "Kiểm tra giá bất thường", "Bảng điều khiển"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    f"Mô hình: XGBoost · {meta['trained_rows']:,} tin đăng\n\n"
    f"MAE {meta['metrics']['MAE_ty']:.2f} tỷ · "
    f"MAPE {meta['metrics']['MAPE_pct']:.1f}% · "
    f"R² {meta['metrics']['R2_log']:.3f}"
)


def _default_index(field: str, preferred: str) -> int:
    """Default a dropdown to the most common real-world value, not the first
    one alphabetically. 93% of listings are 'Đã có sổ' and 84% are 'Nhà ngõ, hẻm'."""
    opts = meta["options"][field]
    return opts.index(preferred) if preferred in opts else 0


def listing_form(key_prefix: str) -> dict:
    """Form nhập thông tin căn nhà, dùng chung cho hai chức năng đầu."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Vị trí**")
        area_key = st.selectbox("Quận / Phường", meta["area_keys"],
                                index=meta["area_keys"].index("Gò Vấp / 12")
                                if "Gò Vấp / 12" in meta["area_keys"] else 0,
                                key=f"{key_prefix}_area")
        streets = meta["streets_by_area"].get(area_key, [])
        street = st.selectbox("Đường", ["(không rõ)"] + streets, key=f"{key_prefix}_street")
        loai_hinh = st.selectbox("Loại hình", meta["options"]["loai_hinh"],
                                 index=_default_index("loai_hinh", "Nhà ngõ, hẻm"),
                                 key=f"{key_prefix}_loai")

    with col2:
        st.markdown("**Diện tích và kết cấu**")
        dien_tich = st.number_input("Diện tích đất (m²)", 10.0, 2000.0, 50.0, 1.0,
                                    key=f"{key_prefix}_dt")
        dt_su_dung = st.number_input("Diện tích sử dụng (m²), 0 = không rõ",
                                     0.0, 6000.0, 100.0, 1.0, key=f"{key_prefix}_dtsd")
        c1, c2 = st.columns(2)
        ngang = c1.number_input("Ngang (m)", 0.0, 100.0, 5.0, 0.1, key=f"{key_prefix}_ng")
        dai = c2.number_input("Dài (m)", 0.0, 200.0, 10.0, 0.1, key=f"{key_prefix}_dai")
        c3, c4, c5 = st.columns(3)
        pn = c3.number_input("Phòng ngủ", 0, 20, 3, key=f"{key_prefix}_pn")
        wc = c4.number_input("WC", 0, 20, 3, key=f"{key_prefix}_wc")
        tang = c5.number_input("Số tầng", 0, 15, 2, key=f"{key_prefix}_tang")

    with col3:
        st.markdown("**Pháp lý và tình trạng**")
        phap_ly = st.selectbox("Giấy tờ pháp lý", meta["options"]["giay_to_phap_ly"],
                               index=_default_index("giay_to_phap_ly", "Đã có sổ"),
                               key=f"{key_prefix}_pl")
        noi_that = st.selectbox("Nội thất", ["(không rõ)"] + meta["options"]["tinh_trang_noi_that"],
                                index=1 + _default_index("tinh_trang_noi_that", "Nội thất đầy đủ"),
                                key=f"{key_prefix}_nt")
        huong = st.selectbox("Hướng cửa chính", ["(không rõ)"] + meta["options"]["huong_cua_chinh"],
                             key=f"{key_prefix}_h")
        dac_diem = st.selectbox("Đặc điểm", ["(không rõ)"] + meta["options"]["dac_diem"],
                                index=1 + _default_index("dac_diem", "Hẻm xe hơi"),
                                key=f"{key_prefix}_dd")

    mo_ta = st.text_area(
        "Mô tả tin đăng (tuỳ chọn — mô hình khai thác các từ khoá như "
        "*hẻm xe hơi*, *hoàn công*, *thang máy*, *1 trệt 2 lầu*)",
        height=90, key=f"{key_prefix}_mt",
    )

    return {
        "area_key": area_key,
        "street": None if street == "(không rõ)" else street,
        "loai_hinh": loai_hinh,
        "dien_tich_m2": dien_tich,
        "dien_tich_su_dung_m2": dt_su_dung or None,
        "chieu_ngang_m": ngang or None,
        "chieu_dai_m": dai or None,
        "so_phong_ngu": pn or None,
        "so_phong_ve_sinh": wc or None,
        "tong_so_tang": tang or None,
        "giay_to_phap_ly": phap_ly,
        "tinh_trang_noi_that": None if noi_that == "(không rõ)" else noi_that,
        "huong_cua_chinh": None if huong == "(không rõ)" else huong,
        "dac_diem": None if dac_diem == "(không rõ)" else dac_diem,
        "tieu_de": "",
        "mo_ta": mo_ta,
    }


# ====================================================== 1. Định giá nhà
if page == "Định giá nhà":
    st.title("Định giá nhà")
    st.caption("Nhập thông tin căn nhà để nhận mức giá gợi ý dựa trên "
               "7 935 tin đăng thực tế tại Bình Thạnh, Gò Vấp và Phú Nhuận.")

    listing = listing_form("price")

    if st.button("Tính giá gợi ý", type="primary", use_container_width=True):
        result = predict_price(listing)
        band = bands[bands["area_key"] == listing["area_key"]].iloc[0]

        st.divider()
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.metric("Giá gợi ý", f"{result['pred_ty']:.2f} tỷ VND",
                  help="Giá trị trung tâm do mô hình XGBoost dự đoán")
        c2.metric("Đơn giá", f"{result['pred_unit_price']:.0f} tr/m²")
        c3.metric("Khoảng tin cậy",
                  f"{result['low_ty']:.2f} – {result['high_ty']:.2f} tỷ",
                  help=f"Dựa trên MAPE {result['segment_mape']:.1f}% "
                       f"đo được ở phân khúc giá này")

        st.info(
            f"**Mặt bằng giá {listing['area_key']}:** trung vị "
            f"{band['p50']:.0f} triệu/m², khoảng phổ biến "
            f"{band['p10']:.0f} – {band['p90']:.0f} triệu/m² "
            f"(dựa trên {int(band['n'])} tin đăng"
            + (", dùng khung cấp quận vì khu vực ít dữ liệu)" if band["fallback"] else ")")
        )

        if result["segment_mape"] > 20:
            st.warning(
                f"Phân khúc giá này có sai số cao ({result['segment_mape']:.1f}%) "
                "do ít dữ liệu huấn luyện. Hãy xem giá gợi ý như một tham khảo thô."
            )

# ============================================ 2. Kiểm tra giá bất thường
elif page == "Kiểm tra giá bất thường":
    st.title("Kiểm tra giá bất thường")
    st.caption("Chấm điểm một tin đăng theo 4 tín hiệu độc lập và đưa ra kết luận.")

    listing = listing_form("anom")
    asking = st.number_input("**Giá rao bán (tỷ VND)**", 0.1, 1000.0, 6.0, 0.1)

    if st.button("Kiểm tra", type="primary", use_container_width=True):
        r = score_anomaly(listing, asking)
        color, icon = VERDICT_STYLE[r["verdict"]]

        st.divider()
        st.markdown(
            f"<div style='padding:18px;border-radius:10px;background:{color}18;"
            f"border-left:5px solid {color}'>"
            f"<div style='font-size:26px;font-weight:700;color:{color}'>{icon} {r['verdict']}</div>"
            f"<div style='color:#52514e;margin-top:6px'>Điểm bất thường "
            f"<b>{r['total_score']:.1f}/100</b> · ngưỡng cảnh báo {r['threshold']:.1f} · "
            f"{r['n_signals']}/4 tín hiệu kích hoạt</div></div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Giá rao", f"{asking:.2f} tỷ", f"{r['unit_price']:.0f} tr/m²")
        c2.metric("Giá mô hình dự đoán", f"{r['pred_ty']:.2f} tỷ")
        delta = (asking - r["pred_ty"]) / r["pred_ty"] * 100
        c3.metric("Chênh lệch", f"{delta:+.1f}%")

        st.subheader("Bốn tín hiệu")
        sig = pd.DataFrame({
            "Tín hiệu": list(r["signals"]),
            "Điểm": list(r["signals"].values()),
            "Trọng số": [meta["weights"][k] for k in
                         ["S_resid", "S_min_max", "S_percentile", "S_ml"]],
        })
        sig["Đóng góp"] = sig["Điểm"] * sig["Trọng số"] * 100

        # A full-width background track per signal. Without it a signal scoring
        # exactly 0 renders as a zero-width bar and its row disappears — but
        # "this signal did not fire" is information the reviewer needs.
        y_enc = alt.Y("Tín hiệu:N", sort=list(r["signals"]), title=None)
        track = (
            alt.Chart(sig.assign(full=1.0))
            .mark_bar(cornerRadius=4, color="#e3e2de", size=22)
            .encode(x=alt.X("full:Q", scale=alt.Scale(domain=[0, 1]),
                            title="Điểm tín hiệu [0–1]"), y=y_enc)
        )
        bars = (
            alt.Chart(sig)
            .mark_bar(cornerRadius=4, color=SERIES[0], size=22)
            .encode(
                x=alt.X("Điểm:Q", scale=alt.Scale(domain=[0, 1])),
                y=y_enc,
                tooltip=["Tín hiệu", alt.Tooltip("Điểm:Q", format=".3f"),
                         alt.Tooltip("Đóng góp:Q", format=".1f", title="Đóng góp (điểm)")],
            )
        )
        st.altair_chart((track + bars).properties(height=170), use_container_width=True)

        b = r["band"]
        st.caption(
            f"Khung tham chiếu {listing['area_key']}: "
            f"P10–P90 = {b['p10']:.0f}–{b['p90']:.0f} tr/m², "
            f"Min/Max = {b['lo']:.0f}–{b['hi']:.0f} tr/m²"
            + ("  ·  khu vực ít dữ liệu, dùng khung cấp quận" if b["fallback"] else "")
        )

        if r["verdict"] != "Bình thường":
            st.warning(
                "**Đây là cảnh báo để con người xem xét, không phải kết luận gian lận.** "
                "Điểm cao có thể do lỗi nhập liệu, do người bán thổi giá, hoặc do đây là "
                "bất động sản thực sự hiếm. Mô hình không phân biệt được ba trường hợp đó."
            )

# =================================================== 3. Bảng điều khiển
else:
    st.title("Bảng điều khiển & giám sát")
    scores = load_scores()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng tin đăng", f"{len(scores):,}")
    c2.metric("Bình thường", f"{(scores['verdict_B'] == 'Bình thường').sum():,}")
    c3.metric("Giá quá rẻ", f"{(scores['verdict_B'] == 'Giá quá rẻ').sum():,}")
    c4.metric("Giá quá đắt", f"{(scores['verdict_B'] == 'Giá quá đắt').sum():,}")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Phân bố điểm bất thường")
        hist = (
            alt.Chart(scores.sample(min(len(scores), 5000), random_state=42))
            .mark_bar(color=SERIES[0])
            .encode(
                x=alt.X("score_B:Q", bin=alt.Bin(maxbins=60), title="Điểm tổng hợp (0–100)"),
                y=alt.Y("count():Q", title="Số tin đăng"),
            )
            .properties(height=260)
        )
        rule = (
            alt.Chart(pd.DataFrame({"x": [meta["score_threshold"]]}))
            .mark_rule(color=SERIES[1], strokeDash=[6, 4], size=2)
            .encode(x="x:Q")
        )
        st.altair_chart(hist + rule, use_container_width=True)
        st.caption(f"Đường đứt nét: ngưỡng cảnh báo top {meta['top_k_pct']:.0f}% "
                   f"= {meta['score_threshold']:.1f} điểm")

    with right:
        st.subheader("Theo quận")
        by_district = (
            scores.assign(district=scores["area_key"].str.split(" / ").str[0])
            .groupby("district")["verdict_B"]
            .value_counts().unstack(fill_value=0).reset_index()
        )
        st.dataframe(by_district, use_container_width=True, hide_index=True)

        st.subheader("Hàng đợi ưu tiên")
        st.markdown(
            f"- **{int((scores['n_signals_B'] >= 3).sum())} tin** bị ≥ 3 tín hiệu gắn cờ\n"
            f"- **{int((scores['S_min_max'] > 0).sum())} tin** vi phạm Min/Max\n"
            f"- **{int((scores['verdict_B'] != 'Bình thường').sum())} tin** trong top "
            f"{meta['top_k_pct']:.0f}%"
        )

    st.divider()
    st.subheader("Danh sách tin đăng cần kiểm duyệt")
    only_flagged = st.checkbox("Chỉ hiện tin bị gắn cờ", value=True)
    view = scores[scores["verdict_B"] != "Bình thường"] if only_flagged else scores
    view = view.nlargest(100, "score_B")[
        ["score_B", "n_signals_B", "verdict_B", "area_key", "street",
         "gia_ban_ty", "don_gia_trieu_m2", "pred_ty", "pct_p10", "pct_p90"]
    ].round(2)
    view.columns = ["Điểm", "Số tín hiệu", "Kết luận", "Khu vực", "Đường",
                    "Giá rao (tỷ)", "Đơn giá", "Giá dự đoán (tỷ)", "P10", "P90"]
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)

    st.divider()
    st.subheader("Giám sát và vòng phản hồi")
    st.markdown(
        """
Bốn nhóm chỉ số cần theo dõi sau khi triển khai:

| Nhóm | Chỉ số | Ngưỡng cảnh báo |
|---|---|---|
| **Chất lượng mô hình** | MAE, MAPE trên tin đăng đã bán | MAPE vượt 20% trong 30 ngày |
| **Trôi dữ liệu** | Trung vị đơn giá từng phường so với lúc huấn luyện | Lệch quá 15% |
| **Khối lượng cảnh báo** | Số tin vào hàng đợi mỗi ngày | Vượt quá năng lực đội kiểm duyệt |
| **Phản hồi kiểm duyệt** | Tỷ lệ tin bị gắn cờ được xác nhận là có vấn đề | Dưới 30% → ngưỡng quá lỏng |

**Vòng phản hồi.** Mỗi quyết định của kiểm duyệt viên (xác nhận có vấn đề / bỏ qua) là một
nhãn thật. Đây chính là thứ Bài toán 2 đang thiếu. Khi đã thu đủ vài nghìn nhãn, bộ trọng số
`w₁..w₄` — hiện được chọn bằng lập luận — nên được **học lại từ dữ liệu** thay vì đặt tay.

**Lịch huấn luyện lại.** Phần EDA cho thấy mặt bằng giá cả ba quận đều tăng trong 4 tháng
gần nhất. Mô hình không có yếu tố thời gian, nên cần huấn luyện lại định kỳ — đề xuất
**hàng tháng**, hoặc ngay khi chỉ số trôi dữ liệu vượt ngưỡng.
        """
    )
