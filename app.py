"""Nhà Tốt: công cụ tra giá nhà và kiểm tra tin đăng.

Ứng dụng dành cho người dùng cuối. Người bán tra xem nhà mình đáng giá bao
nhiêu, người mua kiểm tra xem một tin đăng có bị thổi giá hay không.

Chạy:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from scoring import load_artifacts, predict_price, score_anomaly

ROOT = Path(__file__).parent
FIG = ROOT / "figures"

XANH, CAM, LUC = "#2a78d6", "#eb6834", "#1baf7a"
XANH_NHAT, XANH_DAM = "#9ec5f4", "#184f95"
MUC, XAM, NEN = "#0b0b0b", "#52514e", "#fcfcfb"

KET_LUAN = {
    "Bình thường": (LUC, "✅", "Giá nằm trong mức bình thường của khu vực"),
    "Giá quá rẻ": (XANH, "🔍", "Giá thấp bất thường so với khu vực"),
    "Giá quá đắt": (CAM, "⚠️", "Giá cao bất thường so với khu vực"),
}

st.set_page_config(page_title="Nhà Tốt · Tra giá nhà và kiểm tra tin đăng",
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

# ════════════════════════════════════════════════════ Thanh bên
st.sidebar.image(str(FIG / "nhatot_logo.png"), width=160)
st.sidebar.caption("Tra giá nhà và kiểm tra tin đăng tại Bình Thạnh, "
                   "Gò Vấp và Phú Nhuận")
st.sidebar.divider()

vai = st.sidebar.radio(
    "Bạn đang làm gì?",
    ["Tôi đang bán nhà", "Tôi đang tìm mua nhà"],
)

trang_chinh = ("Định giá nhà của tôi" if vai == "Tôi đang bán nhà"
               else "Kiểm tra một tin đăng")
page = st.sidebar.radio(
    "Chọn mục",
    [trang_chinh, "Mặt bằng giá khu vực", "Về ứng dụng"],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Dựa trên **{meta['trained_rows']:,} tin đăng thật**.\n\n"
    f"Một nửa số lần tra, sai lệch dưới "
    f"**{meta['metrics']['MedAE_ty']:.2f} tỷ**."
)

st.image(str(FIG / "hero_nhatot.png"), use_container_width=True)
st.caption(
    "Đồ án sinh viên, **không liên kết với Nhà Tốt** và không được Nhà Tốt "
    "bảo trợ. Logo thuộc về Nhà Tốt, sử dụng trong phạm vi học tập."
)


# ════════════════════════════════════════════════════ Biểu mẫu
def _mac_dinh(field: str, uu_tien: str) -> int:
    """Chọn sẵn giá trị phổ biến nhất ngoài đời, không phải giá trị đầu bảng.

    93% tin đăng là "Đã có sổ" và 84% là "Nhà ngõ, hẻm", nên để sẵn hai giá
    trị đó giúp đa số người dùng không phải chỉnh gì.
    """
    opts = meta["options"][field]
    return opts.index(uu_tien) if uu_tien in opts else 0


def bieu_mau(khoa: str) -> dict:
    """Biểu mẫu nhập thông tin căn nhà, dùng chung cho cả hai vai."""
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Nhà ở đâu**")
        area_key = st.selectbox(
            "Quận và phường", meta["area_keys"],
            index=meta["area_keys"].index("Gò Vấp / 12")
            if "Gò Vấp / 12" in meta["area_keys"] else 0,
            key=f"{khoa}_area")
        streets = meta["streets_by_area"].get(area_key, [])
        street = st.selectbox("Đường", ["Chưa rõ"] + streets, key=f"{khoa}_street")
        loai_hinh = st.selectbox("Loại nhà", meta["options"]["loai_hinh"],
                                 index=_mac_dinh("loai_hinh", "Nhà ngõ, hẻm"),
                                 key=f"{khoa}_loai")

    with c2:
        st.markdown("**Nhà rộng bao nhiêu**")
        dien_tich = st.number_input("Diện tích đất (m²)", 10.0, 2000.0, 50.0, 1.0,
                                    key=f"{khoa}_dt")
        dt_su_dung = st.number_input("Diện tích sử dụng (m²), để 0 nếu chưa rõ",
                                     0.0, 6000.0, 100.0, 1.0, key=f"{khoa}_dtsd")
        a, b = st.columns(2)
        ngang = a.number_input("Mặt tiền (m)", 0.0, 100.0, 5.0, 0.1,
                               key=f"{khoa}_ng")
        dai = b.number_input("Chiều sâu (m)", 0.0, 200.0, 10.0, 0.1,
                             key=f"{khoa}_dai")
        d, e, f = st.columns(3)
        pn = d.number_input("Phòng ngủ", 0, 20, 3, key=f"{khoa}_pn")
        wc = e.number_input("Nhà vệ sinh", 0, 20, 3, key=f"{khoa}_wc")
        tang = f.number_input("Số tầng", 0, 15, 2, key=f"{khoa}_tang")

    with c3:
        st.markdown("**Giấy tờ và tình trạng**")
        phap_ly = st.selectbox("Giấy tờ", meta["options"]["giay_to_phap_ly"],
                               index=_mac_dinh("giay_to_phap_ly", "Đã có sổ"),
                               key=f"{khoa}_pl")
        noi_that = st.selectbox(
            "Nội thất", ["Chưa rõ"] + meta["options"]["tinh_trang_noi_that"],
            index=1 + _mac_dinh("tinh_trang_noi_that", "Nội thất đầy đủ"),
            key=f"{khoa}_nt")
        huong = st.selectbox("Hướng cửa chính",
                             ["Chưa rõ"] + meta["options"]["huong_cua_chinh"],
                             key=f"{khoa}_h")
        dac_diem = st.selectbox("Đặc điểm",
                                ["Chưa rõ"] + meta["options"]["dac_diem"],
                                index=1 + _mac_dinh("dac_diem", "Hẻm xe hơi"),
                                key=f"{khoa}_dd")

    mo_ta = st.text_area(
        "Mô tả thêm (không bắt buộc). Những từ như *hẻm xe hơi*, *hoàn công*, "
        "*thang máy*, *1 trệt 2 lầu* đều được tính vào kết quả.",
        height=90, key=f"{khoa}_mt")

    return {
        "area_key": area_key,
        "street": None if street == "Chưa rõ" else street,
        "loai_hinh": loai_hinh,
        "dien_tich_m2": dien_tich,
        "dien_tich_su_dung_m2": dt_su_dung or None,
        "chieu_ngang_m": ngang or None,
        "chieu_dai_m": dai or None,
        "so_phong_ngu": pn or None,
        "so_phong_ve_sinh": wc or None,
        "tong_so_tang": tang or None,
        "giay_to_phap_ly": phap_ly,
        "tinh_trang_noi_that": None if noi_that == "Chưa rõ" else noi_that,
        "huong_cua_chinh": None if huong == "Chưa rõ" else huong,
        "dac_diem": None if dac_diem == "Chưa rõ" else dac_diem,
        "tieu_de": "",
        "mo_ta": mo_ta,
    }


def thanh_gia(band: pd.Series, don_gia: float, nhan: str):
    """Đặt một mức đơn giá lên dải giá của phường.

    Một con số đứng riêng không cho biết nó cao hay thấp. Chỉ khi nhìn cạnh
    khoảng giá phổ biến của chính khu vực đó thì người dùng mới đánh giá được.
    """
    khung = pd.DataFrame([{
        "lo": float(band["lo"]), "p10": float(band["p10"]),
        "p50": float(band["p50"]), "p90": float(band["p90"]),
        "hi": float(band["hi"]), "y": "k",
    }])
    diem = pd.DataFrame([{"x": don_gia, "y": "k", "nhan": nhan}])
    truc_y = alt.Y("y:N", title=None, axis=None)

    day = alt.Chart(khung).mark_bar(color="#e8eef6", height=36).encode(
        x=alt.X("lo:Q", title="Đơn giá (triệu đồng mỗi m²)"), x2="hi:Q", y=truc_y)
    giua = alt.Chart(khung).mark_bar(color=XANH_NHAT, height=36).encode(
        x="p10:Q", x2="p90:Q", y=truc_y)
    trung_vi = alt.Chart(khung).mark_tick(
        color=XANH_DAM, thickness=3, size=42).encode(x="p50:Q", y=truc_y)
    cham = alt.Chart(diem).mark_point(
        shape="diamond", size=340, filled=True, color=CAM,
        stroke="#ffffff", strokeWidth=1.8).encode(
        x="x:Q", y=truc_y,
        tooltip=[alt.Tooltip("nhan:N", title="Mức giá")])
    # Nhãn đặt bên phải hình thoi. Đặt phía trên thì nó tràn ra ngoài vùng vẽ.
    chu = alt.Chart(diem).mark_text(
        align="left", dx=15, fontSize=12, fontWeight="bold", color=CAM).encode(
        x="x:Q", y=truc_y, text="nhan:N")

    st.altair_chart((day + giua + trung_vi + cham + chu).properties(height=112),
                    use_container_width=True)
    st.caption(
        "Dải nhạt là toàn bộ khoảng giá đã ghi nhận ở phường này. Dải đậm là "
        "khoảng phổ biến, tức 8 trên 10 căn rơi vào đó. Vạch dọc là mức giá "
        "giữa. Hình thoi cam là mức giá đang xét."
    )


# ══════════════════════════════════════════ 1a. Người bán định giá
if page == "Định giá nhà của tôi":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("Nhà của bạn đáng giá bao nhiêu?")
        st.write(
            "Điền thông tin căn nhà, ứng dụng sẽ đưa ra mức giá tham khảo dựa "
            f"trên **{meta['trained_rows']:,} tin đăng thật** tại Bình Thạnh, "
            "Gò Vấp và Phú Nhuận. Bạn có thể dùng mức này làm điểm xuất phát "
            "khi đặt giá rao."
        )
    with c2:
        st.image(str(FIG / "vai_nguoi_ban.png"), use_container_width=True)

    listing = bieu_mau("ban")

    if st.button("Xem giá tham khảo", type="primary", use_container_width=True):
        kq = predict_price(listing)
        band = bands[bands["area_key"] == listing["area_key"]].iloc[0]
        st.session_state["ban_kq"] = (kq, listing["area_key"])

    if "ban_kq" in st.session_state:
        kq, akey = st.session_state["ban_kq"]
        band = bands[bands["area_key"] == akey].iloc[0]

        st.divider()
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.metric("Giá tham khảo", f"{kq['pred_ty']:.2f} tỷ")
        c2.metric("Tính ra mỗi m²", f"{kq['pred_unit_price']:.0f} triệu")
        c3.metric("Khoảng hợp lý",
                  f"{kq['low_ty']:.2f} tới {kq['high_ty']:.2f} tỷ")

        st.write(
            f"Với những căn tương tự ở **{akey}**, mức giá thường thấy là "
            f"khoảng **{band['p50']:.0f} triệu mỗi m²**, và phần lớn nằm trong "
            f"khoảng {band['p10']:.0f} tới {band['p90']:.0f} triệu. "
            f"Con số này tính từ {int(band['n'])} tin đăng"
            + (" của cả quận, vì phường của bạn có ít dữ liệu."
               if band["fallback"] else " ngay tại phường của bạn.")
        )
        thanh_gia(band, kq["pred_unit_price"], f"{kq['pred_unit_price']:.0f} tr/m²")

        st.subheader("Bạn đang định rao bao nhiêu?")
        st.caption("Nhập mức giá bạn dự định đăng để xem nó nằm ở đâu so với "
                   "thị trường.")
        rao = st.number_input("Giá bạn định rao (tỷ đồng)", 0.1, 1000.0,
                              float(round(kq["pred_ty"], 2)), 0.1, key="ban_rao")
        lech = (rao - kq["pred_ty"]) / kq["pred_ty"] * 100

        # Khi lệch gần bằng không thì "+0%" hay "-0%" đọc rất kỳ, nên tách
        # riêng câu chữ cho trường hợp đó.
        if abs(lech) < 1:
            st.success(
                f"Mức **{rao:.2f} tỷ** gần như trùng với giá tham khảo. "
                "Tin đăng của bạn nhiều khả năng được người mua xem là "
                "nghiêm túc."
            )
        elif abs(lech) <= 10:
            st.success(
                f"Mức **{rao:.2f} tỷ** chênh {lech:+.0f}% so với giá tham khảo. "
                "Đây là khoảng hợp lý, tin đăng của bạn nhiều khả năng được "
                "người mua xem là nghiêm túc."
            )
        elif lech > 10:
            st.warning(
                f"Mức **{rao:.2f} tỷ** cao hơn giá tham khảo {lech:.0f}%. "
                "Giá cao hơn thị trường thường khiến tin đăng lâu có người hỏi. "
                "Nếu nhà bạn có điểm đặc biệt chưa khai trong biểu mẫu, hãy ghi "
                "rõ vào phần mô tả khi đăng tin."
            )
        else:
            st.info(
                f"Mức **{rao:.2f} tỷ** thấp hơn giá tham khảo {abs(lech):.0f}%. "
                "Bạn có thể bán nhanh hơn, nhưng hãy cân nhắc kỹ vì mức này "
                "nằm dưới mặt bằng chung của khu vực."
            )

        if kq["segment_mape"] > 20:
            st.caption(
                f"Lưu ý: phân khúc giá này có ít dữ liệu nên độ chính xác thấp "
                f"hơn bình thường. Hãy xem con số như một tham khảo thô và đối "
                f"chiếu thêm với các tin đăng đang có trong khu vực."
            )

# ══════════════════════════════════════════ 1b. Người mua kiểm tra
elif page == "Kiểm tra một tin đăng":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("Tin đăng này có đáng tin không?")
        st.write(
            "Thấy một căn ưng ý nhưng không chắc giá có hợp lý? Nhập thông tin "
            "trong tin đăng cùng mức giá người bán đưa ra, ứng dụng sẽ so sánh "
            "với mặt bằng thực tế của khu vực và cho biết mức giá đó có bất "
            "thường hay không."
        )
    with c2:
        st.image(str(FIG / "vai_nguoi_mua.png"), use_container_width=True)

    listing = bieu_mau("mua")
    rao = st.number_input("**Giá người bán đang rao (tỷ đồng)**",
                          0.1, 1000.0, 6.0, 0.1)

    if st.button("Kiểm tra giá", type="primary", use_container_width=True):
        r = score_anomaly(listing, rao)
        mau, icon, tom_tat = KET_LUAN[r["verdict"]]

        st.divider()
        st.markdown(
            f"<div style='padding:20px;border-radius:12px;background:{mau}18;"
            f"border-left:6px solid {mau}'>"
            f"<div style='font-size:25px;font-weight:700;color:{mau}'>"
            f"{icon} {tom_tat}</div></div>",
            unsafe_allow_html=True,
        )

        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Người bán rao", f"{rao:.2f} tỷ",
                  f"{r['unit_price']:.0f} triệu mỗi m²")
        c2.metric("Mức giá hợp lý theo ứng dụng", f"{r['pred_ty']:.2f} tỷ")
        lech = (rao - r["pred_ty"]) / r["pred_ty"] * 100
        c3.metric("Chênh lệch", f"{lech:+.0f}%")

        band = bands[bands["area_key"] == listing["area_key"]].iloc[0]
        thanh_gia(band, r["unit_price"], f"{r['unit_price']:.0f} tr/m²")

        if r["verdict"] == "Giá quá đắt":
            st.warning(
                "**Bạn nên làm gì tiếp theo.** Hỏi người bán vì sao giá cao hơn "
                "mặt bằng khu vực. Có thể căn nhà có điểm đặc biệt chưa thể "
                "hiện trong tin đăng, chẳng hạn vừa sửa lớn, vị trí góc hai mặt "
                "tiền, hoặc đã hoàn công. Nếu không có lý do rõ ràng, hãy so "
                "sánh thêm vài tin đăng khác cùng phường trước khi đặt cọc."
            )
        elif r["verdict"] == "Giá quá rẻ":
            st.info(
                "**Bạn nên làm gì tiếp theo.** Giá thấp bất thường không hẳn là "
                "cơ hội tốt. Hãy kiểm tra kỹ giấy tờ pháp lý, tình trạng tranh "
                "chấp, quy hoạch, và diện tích thực tế so với diện tích ghi "
                "trong tin. Cũng có khả năng người bán gõ nhầm số."
            )
        else:
            st.success(
                "**Bạn nên làm gì tiếp theo.** Mức giá này phù hợp với mặt bằng "
                "khu vực. Việc còn lại là kiểm tra giấy tờ pháp lý và xem nhà "
                "trực tiếp, hai thứ mà ứng dụng không thể thay bạn làm."
            )

        with st.expander("Vì sao ứng dụng kết luận như vậy?"):
            # Khoá do scoring.py trả về là tên kỹ thuật. Đổi sang câu chữ
            # người thường hiểu được, và giữ nguyên thứ tự gốc.
            ten_tin_hieu = {
                "Residual-Z": "Chênh lệch so với mức giá ứng dụng ước tính",
                "Min/Max": "Vượt ra ngoài khoảng giá từng ghi nhận ở phường",
                "P10–P90": "Vị trí so với các căn khác cùng phường",
                "Isolation Forest": "Tổng thể căn nhà khác thường so với phần còn lại",
            }
            sig = pd.DataFrame({
                "Dấu hiệu": [ten_tin_hieu.get(k, k) for k in r["signals"]],
                "Mức độ": list(r["signals"].values()),
            })
            nen_track = alt.Chart(sig.assign(full=1.0)).mark_bar(
                cornerRadius=4, color="#e8eef6", size=24).encode(
                x=alt.X("full:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"), title="Mức độ bất thường"),
                y=alt.Y("Dấu hiệu:N", sort=list(sig["Dấu hiệu"]), title=None,
                        axis=alt.Axis(labelLimit=340, labelOverlap=False)))
            cot = alt.Chart(sig).mark_bar(
                cornerRadius=4, color=XANH, size=24).encode(
                x=alt.X("Mức độ:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Dấu hiệu:N", sort=list(sig["Dấu hiệu"]), title=None),
                tooltip=["Dấu hiệu", alt.Tooltip("Mức độ:Q", format=".0%")])
            st.altair_chart((nen_track + cot).properties(height=190),
                            use_container_width=True)
            st.write(
                f"Ứng dụng xét bốn dấu hiệu độc lập. Lần này có "
                f"**{r['n_signals']} trên 4 dấu hiệu** cho thấy bất thường. "
                "Thanh càng dài nghĩa là dấu hiệu đó càng rõ."
            )

        st.caption(
            "Kết quả này là gợi ý để bạn xem xét kỹ hơn, không phải kết luận "
            "về việc người bán có gian dối hay không. Giá lệch mặt bằng có thể "
            "do nhiều nguyên nhân hợp lý."
        )

# ══════════════════════════════════════ 2. Mặt bằng giá khu vực
elif page == "Mặt bằng giá khu vực":
    st.title("Giá nhà ở từng khu vực")
    st.write(
        "Xem mức giá đang phổ biến ở từng phường để biết khu nào vừa túi tiền, "
        "khu nào đang tăng giá."
    )

    quan = st.selectbox("Chọn quận", sorted(bands["district"].dropna().unique()))
    b_quan = bands[bands["district"] == quan].sort_values("p50", ascending=False)
    b_quan = b_quan[~b_quan["fallback"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Số phường có dữ liệu", f"{len(b_quan)}")
    c2.metric("Giá giữa của quận", f"{b_quan['p50'].median():.0f} triệu mỗi m²")
    c3.metric("Số tin đăng", f"{int(b_quan['n'].sum()):,}")

    st.subheader(f"Các phường ở {quan}, xếp theo mức giá")
    ve = b_quan.assign(ten=b_quan["area_key"].str.split(" / ").str[1])
    dai = alt.Chart(ve).mark_bar(color="#e8eef6", height=15).encode(
        x=alt.X("p10:Q", title="Đơn giá (triệu đồng mỗi m²)",
                axis=alt.Axis(tickCount=8)), x2="p90:Q",
        y=alt.Y("ten:N", sort=None, title="Phường",
                axis=alt.Axis(labelOverlap=False)))
    cham = alt.Chart(ve).mark_point(
        shape="diamond", size=110, filled=True, color=XANH).encode(
        x="p50:Q", y=alt.Y("ten:N", sort=None, title="Phường"),
        tooltip=[alt.Tooltip("ten:N", title="Phường"),
                 alt.Tooltip("p50:Q", format=".0f", title="Giá giữa"),
                 alt.Tooltip("p10:Q", format=".0f", title="Mức thấp"),
                 alt.Tooltip("p90:Q", format=".0f", title="Mức cao"),
                 alt.Tooltip("n:Q", format=",.0f", title="Số tin")])
    st.altair_chart((dai + cham).properties(height=max(220, 26 * len(ve))),
                    use_container_width=True)
    st.caption("Mỗi thanh là khoảng giá phổ biến của một phường, hình thoi là "
               "mức giá giữa. Di chuột lên hình thoi để xem số liệu chi tiết.")

    st.subheader("Khu vực nào đang tăng giá?")
    xu_huong = b_quan.assign(
        ten=b_quan["area_key"].str.split(" / ").str[1],
        thay_doi=(b_quan["price_hist_last"] - b_quan["price_hist_mean"])
        / b_quan["price_hist_mean"] * 100,
    ).dropna(subset=["thay_doi"]).sort_values("thay_doi")
    st.altair_chart(
        alt.Chart(xu_huong).mark_bar(cornerRadius=3).encode(
            x=alt.X("thay_doi:Q", title="Chênh lệch so với mức trung bình (%)",
                    axis=alt.Axis(tickCount=7)),
            y=alt.Y("ten:N", sort=None, title="Phường",
                    axis=alt.Axis(labelOverlap=False)),
            color=alt.condition(alt.datum.thay_doi > 0, alt.value(CAM),
                                alt.value(XANH)),
            tooltip=[alt.Tooltip("ten:N", title="Phường"),
                     alt.Tooltip("thay_doi:Q", format="+.1f", title="Chênh (%)")],
        ).properties(height=max(220, 26 * len(xu_huong))),
        use_container_width=True)
    st.caption(
        "Cột cam là phường có mức giá gần đây **cao hơn** trung bình của chính "
        "nó, cột xanh là **thấp hơn**. Đây là tín hiệu tham khảo về xu hướng "
        "ngắn hạn, không phải dự báo giá tương lai."
    )

    st.divider()
    st.subheader("So sánh ba quận")
    sc = load_scores()
    sc = sc.assign(quan=sc["area_key"].str.split(" / ").str[0])

    st.altair_chart(
        alt.Chart(sc.sample(min(len(sc), 4000), random_state=42))
        .mark_circle(size=22, opacity=0.28, color=XANH)
        .encode(
            x=alt.X("don_gia_trieu_m2:Q", title="Đơn giá (triệu đồng mỗi m²)",
                    scale=alt.Scale(domain=[0, 400], clamp=True),
                    axis=alt.Axis(tickCount=9)),
            y=alt.Y("quan:N", title=None,
                    axis=alt.Axis(labelFontSize=12, labelOverlap=False)),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("area_key:N", title="Phường"),
                     alt.Tooltip("don_gia_trieu_m2:Q", format=".0f",
                                 title="Đơn giá"),
                     alt.Tooltip("gia_ban_ty:Q", format=".2f",
                                 title="Giá rao (tỷ)")],
        )
        .transform_calculate(jitter="random()")
        .properties(height=240).interactive(),
        use_container_width=True)
    st.caption(
        "Mỗi chấm là một tin đăng. Điều đáng chú ý là **chênh lệch bên trong "
        "cùng một quận còn lớn hơn chênh lệch giữa ba quận**, nên khi so giá "
        "thì nên so ở cấp phường chứ đừng so cấp quận. Cuộn để phóng to, kéo "
        "để di chuyển."
    )

    st.divider()
    st.subheader("Tra giá theo con đường")
    st.write("Chọn phường rồi chọn đường để xem mặt bằng giá ngay tại đó.")

    c1, c2 = st.columns(2)
    phuong = c1.selectbox("Phường",
                          sorted(sc[sc["quan"] == quan]["area_key"].unique()))
    tin_phuong = sc[sc["area_key"] == phuong]
    # Chỉ liệt kê đường có đủ tin đăng, dưới ba tin thì trung vị vô nghĩa
    dem = tin_phuong["street"].value_counts()
    duong_du = sorted(dem[dem >= 3].index.tolist())

    if not duong_du:
        c2.info("Phường này chưa có con đường nào đủ 3 tin đăng để thống kê.")
    else:
        duong = c2.selectbox(f"Đường, {len(duong_du)} đường có đủ dữ liệu",
                             duong_du)
        td = tin_phuong[tin_phuong["street"] == duong]
        tv_phuong = tin_phuong["don_gia_trieu_m2"].median()
        tv_duong = td["don_gia_trieu_m2"].median()
        lech = (tv_duong - tv_phuong) / tv_phuong * 100

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Đơn giá giữa của đường", f"{tv_duong:.0f} triệu/m²")
        d2.metric("So với cả phường", f"{lech:+.0f}%")
        d3.metric("Số tin đăng", f"{len(td)}")
        d4.metric("Giá rao giữa", f"{td['gia_ban_ty'].median():.2f} tỷ")

        if len(td) < 8:
            st.caption(
                f"Chỉ {len(td)} tin đăng trên đường này, nên con số mang tính "
                "tham khảo. Càng ít tin thì trung vị càng dễ bị một căn bất "
                "thường kéo lệch."
            )

        st.altair_chart(
            alt.Chart(td).mark_circle(size=90, opacity=0.75, color=CAM).encode(
                x=alt.X("don_gia_trieu_m2:Q", title="Đơn giá (triệu đồng mỗi m²)",
                        axis=alt.Axis(tickCount=7)),
                y=alt.Y("gia_ban_ty:Q", title="Giá rao (tỷ đồng)",
                        axis=alt.Axis(tickCount=6)),
                tooltip=[alt.Tooltip("gia_ban_ty:Q", format=".2f",
                                     title="Giá rao (tỷ)"),
                         alt.Tooltip("don_gia_trieu_m2:Q", format=".0f",
                                     title="Đơn giá"),
                         alt.Tooltip("verdict_B:N", title="Đánh giá")],
            ).properties(height=260).interactive(), use_container_width=True)
        st.caption(f"Toàn bộ {len(td)} tin đăng ghi nhận trên {duong}. "
                   "Rê chuột lên từng chấm để xem chi tiết.")

    st.divider()
    st.subheader("Có bao nhiêu nhà trong tầm tiền của bạn?")
    tam = st.slider("Khoảng giá bạn cân nhắc (tỷ đồng)", 0.5, 40.0,
                    (3.0, 8.0), 0.5)
    trong_tam = sc[(sc["gia_ban_ty"] >= tam[0]) & (sc["gia_ban_ty"] <= tam[1])]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.altair_chart(
            alt.Chart(trong_tam).mark_bar(cornerRadius=2, color=XANH).encode(
                x=alt.X("gia_ban_ty:Q", bin=alt.Bin(maxbins=40),
                        title="Giá rao (tỷ đồng)",
                        axis=alt.Axis(tickCount=8)),
                y=alt.Y("count():Q", title="Số tin đăng"),
                tooltip=[alt.Tooltip("count():Q", title="Số tin đăng")],
            ).properties(height=260), use_container_width=True)
    with c2:
        st.metric("Tin đăng trong tầm giá", f"{len(trong_tam):,}")
        st.metric("Chiếm tỷ lệ", f"{len(trong_tam) / len(sc) * 100:.0f}%")
        st.caption("trên tổng số tin đăng của cả ba quận")

    theo_quan = (trong_tam.groupby("quan").size()
                 .rename("so_tin").reset_index()
                 .sort_values("so_tin", ascending=False))
    st.altair_chart(
        alt.Chart(theo_quan).mark_bar(cornerRadius=3, color=XANH_DAM).encode(
            x=alt.X("so_tin:Q", title="Số tin đăng trong tầm giá",
                    axis=alt.Axis(tickCount=6)),
            y=alt.Y("quan:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("quan:N", title="Quận"),
                     alt.Tooltip("so_tin:Q", title="Số tin")],
        ).properties(height=130), use_container_width=True)
    st.caption("Quận nào có nhiều lựa chọn nhất trong khoảng tiền bạn chọn.")

# ══════════════════════════════════════════════ 3. Về ứng dụng
else:
    st.title("Về ứng dụng")

    st.header("Ứng dụng này làm được gì")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**Nếu bạn đang bán nhà**

Điền thông tin căn nhà để nhận một mức giá tham khảo, kèm khoảng giá hợp lý.
Bạn cũng có thể nhập mức giá định rao để xem nó nằm ở đâu so với thị trường.
            """
        )
    with c2:
        st.markdown(
            """
**Nếu bạn đang tìm mua**

Nhập thông tin một tin đăng cùng giá người bán đưa ra. Ứng dụng so sánh với
mặt bằng khu vực và cho biết mức giá đó có bất thường hay không, kèm gợi ý
nên kiểm tra thêm điều gì.
            """
        )

    st.divider()
    st.header("Số liệu đến từ đâu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tin đăng đã học", f"{meta['trained_rows']:,}")
    c1.caption("tại Bình Thạnh, Gò Vấp, Phú Nhuận")
    c2.metric("Sai lệch thường gặp", f"{meta['metrics']['MedAE_ty']:.2f} tỷ")
    c2.caption("một nửa số lần tra sai dưới mức này")
    c3.metric("Sai lệch trung bình", f"{meta['metrics']['MAPE_pct']:.0f}%")
    c3.caption("tính theo phần trăm giá trị căn nhà")

    st.divider()
    st.header("Khi nào không nên tin kết quả")
    st.markdown(
        """
- **Nhà quá rẻ hoặc quá đắt so với mặt bằng.** Ứng dụng đoán chính xác nhất ở
  phân khúc tầm trung. Với nhà vài trăm triệu hoặc trên vài chục tỷ, sai số
  lớn hơn nhiều vì có rất ít tin đăng để học.
- **Phường có ít dữ liệu.** Khi đó ứng dụng mượn khung giá của cả quận, và sẽ
  nói rõ điều này trong kết quả.
- **Những thứ ứng dụng không nhìn thấy.** Tình trạng thực tế của căn nhà,
  tranh chấp pháp lý, quy hoạch, hàng xóm, hướng nắng gió. Không có con số nào
  thay được việc đi xem nhà và kiểm tra giấy tờ.
- **Dữ liệu là giá rao, không phải giá bán thật.** Giá chốt thường thấp hơn
  giá rao, nên hãy xem kết quả như mặt bằng rao bán của khu vực.
        """
    )

    st.divider()
    with st.expander("Chi tiết kỹ thuật, dành cho người quan tâm"):
        st.markdown(
            f"""
**Dự đoán giá.** Mô hình XGBoost học trên {meta['trained_rows']:,} tin đăng đã
làm sạch, huấn luyện trên thang logarit của giá vì phân bố giá lệch phải rất
mạnh. Sai số trên tập kiểm tra: MAE {meta['metrics']['MAE_ty']:.2f} tỷ,
MedAE {meta['metrics']['MedAE_ty']:.2f} tỷ, MAPE
{meta['metrics']['MAPE_pct']:.1f} phần trăm, R² trên thang log
{meta['metrics']['R2_log']:.3f}.

**Phát hiện giá bất thường.** Bốn tín hiệu độc lập được tổ hợp: phần dư của mô
hình giá, vi phạm khung giá nhỏ nhất và lớn nhất của phường, vị trí phân vị, và
Isolation Forest. Bài toán này **không có nhãn đúng**, nên trọng số tổ hợp được
đặt theo lập luận chứ không học từ dữ liệu, và ngưỡng cảnh báo lấy ở mức
{meta['top_k_pct']:.0f} phần trăm cao nhất.
            """
        )
        t1, t2 = st.tabs(["Mô hình giá", "Phát hiện bất thường"])
        with t1:
            st.image(str(FIG / "10_model_comparison.png"), use_container_width=True)
            st.image(str(FIG / "12_feature_importance.png"), use_container_width=True)
            st.image(str(FIG / "08_market_curve.png"), use_container_width=True)
            st.caption(
                "Mặt bằng giá cả ba quận đều nhích lên trong bốn tháng cuối "
                "của dữ liệu. Mô hình không có yếu tố thời gian nên cần huấn "
                "luyện lại định kỳ. Biểu đồ này để dạng ảnh vì dữ liệu đã làm "
                "sạch không còn giữ cột ngày đăng."
            )
            st.image(str(FIG / "13_error_by_segment.png"), use_container_width=True)
            st.caption(
                "Sai số theo phân khúc giá có dạng chữ U: nhà rẻ và nhà đắt đều "
                "khó đoán hơn nhà tầm trung. Dạng này đã được kiểm chứng trên ba "
                "cách chia nhóm khác nhau."
            )
        with t2:
            st.image(str(FIG / "16_score_distribution.png"), use_container_width=True)
            st.image(str(FIG / "17_signal_agreement.png"), use_container_width=True)
            st.image(str(FIG / "18_validation.png"), use_container_width=True)
            st.caption(
                "Phép kiểm chứng gián tiếp: nhóm tin bị gắn cờ có phân vị giá "
                "trung bình 68,1 so với 47,1 của nhóm bình thường, đúng hướng "
                "kỳ vọng."
            )

        scores = load_scores()
        st.write("")
        st.write("**Kết quả chấm toàn bộ kho tin đăng**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Tổng tin", f"{len(scores):,}")
        d2.metric("Bình thường", f"{(scores['verdict_B'] == 'Bình thường').sum():,}")
        d3.metric("Giá quá rẻ", f"{(scores['verdict_B'] == 'Giá quá rẻ').sum():,}")
        d4.metric("Giá quá đắt", f"{(scores['verdict_B'] == 'Giá quá đắt').sum():,}")

    st.divider()
    st.caption(
        "Đồ án tốt nghiệp Data Science, Nhóm 5: Lư Trần Nguyên Đăng, "
        "Nguyễn Anh Duy, Phạm Thị Phương Thúy. Trung Tâm Tin Học, Trường Đại "
        "học Khoa học Tự nhiên, ĐHQG-HCM.\n\n"
        "Dữ liệu chỉ dùng cho mục đích học tập, không chia sẻ hoặc sử dụng cho "
        "mục đích khác."
    )
