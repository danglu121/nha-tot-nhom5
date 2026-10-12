"""Nhà Tốt: công cụ tra giá nhà và tìm tin đăng.

Ứng dụng dành cho người dùng cuối. Người bán tra xem nhà mình đáng giá bao
nhiêu, người mua mô tả căn nhà mình muốn rồi nhận về những tin đăng thật gần
với mô tả nhất, mỗi tin kèm nhãn cho biết giá có bất thường hay không.

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

st.set_page_config(page_title="Nhà Tốt · Tra giá nhà và tìm nhà",
                   page_icon="🏠", layout="wide")


@st.cache_resource
def artifacts():
    return load_artifacts()


@st.cache_data
def load_scores():
    return pd.read_parquet(ROOT / "data" / "anomaly_scores.parquet")


@st.cache_data
def load_hang_doi():
    """397 tin bị gắn cờ, kèm tiêu đề và mô tả để quản trị viên xem xét."""
    return pd.read_parquet(ROOT / "data" / "hang_doi_duyet.parquet")


@st.cache_data
def load_tin_dang():
    """Toàn bộ kho tin đăng cho trang tìm nhà, kèm nhãn giá đã chấm sẵn."""
    return pd.read_parquet(ROOT / "data" / "tin_dang.parquet")


art = artifacts()
meta = art["meta"]
bands = art["bands"]

# ════════════════════════════════════════════════════ Thanh bên
st.sidebar.image(str(FIG / "nhatot_logo.png"), width=160)
st.sidebar.caption("Tra giá nhà và tìm tin đăng tại Bình Thạnh, "
                   "Gò Vấp và Phú Nhuận")
st.sidebar.divider()

vai = st.sidebar.radio(
    "Bạn đang làm gì?",
    ["Tôi đang bán nhà", "Tôi đang tìm mua nhà", "Tôi là quản trị viên"],
)

if vai == "Tôi đang bán nhà":
    trang_chinh = "Định giá nhà của tôi"
elif vai == "Tôi đang tìm mua nhà":
    trang_chinh = "Tìm nhà phù hợp"
else:
    trang_chinh = "Duyệt tin đăng"

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

# Thông tin nhóm đặt cuối thanh bên, dưới phần điều hướng, để không chen
# vào luồng thao tác của người dùng.
st.sidebar.divider()
st.sidebar.markdown("**Thông tin liên hệ**")
for _ten, _mail in [
    ("Lư Trần Nguyên Đăng", "lutrannguyendang@gmail.com"),
    ("Phạm Thị Phương Thúy", "phuongthuy219@gmail.com"),
    ("Nguyễn Anh Duy", "toulouse18uef@gmail.com"),
]:
    st.sidebar.markdown(
        f"<div style='line-height:1.35;margin-bottom:9px'>"
        f"<span style='font-size:13px'>{_ten}</span><br>"
        f"<a href='mailto:{_mail}' style='font-size:11.5px;color:{XAM};"
        f"text-decoration:none'>{_mail}</a></div>",
        unsafe_allow_html=True)
st.sidebar.caption(
    "Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM")

st.image(str(FIG / "hero_nhatot.png"), use_container_width=True)
st.caption(
    "Đồ án sinh viên, **không liên kết với Nhà Tốt** và không được Nhà Tốt "
    "bảo trợ. Logo thuộc về Nhà Tốt, sử dụng trong phạm vi học tập."
)


# ════════════════════════════════════════════════════ Biểu mẫu
def zoom_giu_shift():
    """Phóng to bằng con lăn, nhưng phải giữ Shift.

    Để .interactive() mặc định thì biểu đồ nuốt mọi thao tác con lăn: người
    dùng tưởng đang cuộn trang mà hoá ra đang phóng to biểu đồ, và trục bị
    kéo ra những khoảng vô nghĩa. Buộc giữ Shift thì cuộn trang vẫn bình
    thường, ai muốn phóng to thì vẫn phóng được.
    """
    return alt.selection_interval(bind="scales",
                                  zoom="wheel![event.shiftKey]")


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


# ══════════════════════════════════════ Kho tin đăng cho người mua
GIA_SAN, GIA_TRAN = 0.5, 40.0

PHAM_VI = ["Phường tôi chọn", "Cả quận", "Cả ba quận"]

MAC_DINH_LOC = {
    "loc_pham_vi": "Cả quận",
    "loc_tam_gia": (GIA_SAN, GIA_TRAN),
    "loc_pn": 0,
    "loc_dt": 0,
    "loc_an_toan": False,
    "loc_sap_xep": "Phù hợp nhất",
    "loc_so_tin": 8,
}


def tam_gia_mac_dinh(gia_uoc: float) -> tuple[float, float]:
    """Tầm giá mở sẵn quanh mức ứng dụng ước tính cho căn người mua mô tả."""
    return (float(max(GIA_SAN, round(gia_uoc * 0.60 * 2) / 2)),
            float(min(GIA_TRAN, round(gia_uoc * 1.40 * 2) / 2)))

THU_TU_SAP_XEP = {
    "Phù hợp nhất": ("phu_hop", False),
    "Giá thấp trước": ("gia_ban_ty", True),
    "Giá cao trước": ("gia_ban_ty", False),
    "Đơn giá rẻ nhất": ("don_gia_trieu_m2", True),
}

NHAN_NGAN = {
    "Bình thường": ("Giá bình thường", LUC),
    "Giá quá rẻ": ("Giá thấp bất thường", XANH),
    "Giá quá đắt": ("Giá cao bất thường", CAM),
}

TRONG_SO_PHU_HOP = {"vi_tri": 0.30, "dien_tich": 0.22, "gia": 0.18,
                    "phong_ngu": 0.15, "loai_nha": 0.10, "giay_to": 0.05}

TEN_TIN_HIEU = {
    "s_resid": "Chênh lệch so với mức giá ứng dụng ước tính",
    "s_min_max": "Vượt ra ngoài khoảng giá từng ghi nhận ở phường",
    "s_pct": "Vị trí so với các căn khác cùng phường",
    "s_ml": "Tổng thể căn nhà khác thường so với phần còn lại",
}


def cham_phu_hop(kho: pd.DataFrame, yc: dict, gia_uoc: float) -> np.ndarray:
    """Chấm độ phù hợp 0 tới 1 của từng tin với căn nhà người mua mô tả.

    Người mua không phải nhập giá. Mốc giá dùng để so sánh chính là mức mà mô
    hình ước tính cho căn họ vừa mô tả, nên danh sách vẫn bám sát tầm tiền hợp
    lý của căn đó mà không bắt người mua tự định giá trước.
    """
    def gan_nhau(cot: pd.Series, muc: float, gap: float) -> np.ndarray:
        """Bằng 1 khi trùng mức, giảm dần về 0 khi lệch tới `gap` lần."""
        x = np.maximum(cot.to_numpy(dtype=float), 1e-6)
        return np.clip(1 - np.abs(np.log(x / max(muc, 1e-6))) / np.log(gap), 0, 1)

    quan = yc["area_key"].split(" / ")[0]
    cung_phuong = kho["area_key"].astype(str).to_numpy() == yc["area_key"]
    cung_duong = kho["street"].astype(str).to_numpy() == (yc["street"] or "\x00")
    vi_tri = np.where(
        cung_phuong & cung_duong, 1.00,
        np.where(cung_phuong, 0.90,
                 np.where(kho["district"].astype(str).to_numpy() == quan,
                          0.50, 0.15)))

    dien_tich = gan_nhau(kho["dien_tich_m2"], float(yc["dien_tich_m2"]), 2.0)
    gia = gan_nhau(kho["gia_ban_ty"], float(gia_uoc), 1.7)

    pn = kho["so_phong_ngu"].to_numpy(dtype=float)
    pn_yc = yc.get("so_phong_ngu")
    # Thiếu số phòng ngủ thì cho điểm giữa, vì không có cơ sở để thưởng hay
    # phạt tin đó so với những tin khai đầy đủ.
    phong_ngu = (np.where(np.isnan(pn), 0.5,
                          1 - np.clip(np.abs(pn - pn_yc) / 3, 0, 1))
                 if pn_yc else np.full(len(kho), 0.5))

    loai_nha = np.where(
        kho["loai_hinh"].astype(str).to_numpy() == yc["loai_hinh"], 1.0, 0.25)
    gt = kho["giay_to_phap_ly"].astype(str).to_numpy()
    giay_to = np.where(gt == str(yc["giay_to_phap_ly"]), 1.0,
                       np.where(gt == "nan", 0.5, 0.40))

    w = TRONG_SO_PHU_HOP
    return (w["vi_tri"] * vi_tri + w["dien_tich"] * dien_tich
            + w["gia"] * gia + w["phong_ngu"] * phong_ngu
            + w["loai_nha"] * loai_nha + w["giay_to"] * giay_to)


def bieu_do_goi_y(ds: pd.DataFrame, kq: dict, yc: dict):
    """Đặt các tin gợi ý lên mặt phẳng diện tích và giá.

    Một danh sách dọc không cho thấy các lựa chọn đắt rẻ ra sao so với nhau.
    Biểu đồ này cho thấy ngay, và dấu thoi đen là căn người mua mô tả.
    """
    ds = ds.assign(
        nhan=[NHAN_NGAN[v][0] for v in ds["verdict"].astype(str)],
        phu_hop_pct=(ds["phu_hop"] * 100).round(0),
    )
    thu_tu = [NHAN_NGAN[k][0] for k in KET_LUAN]
    diem = alt.Chart(ds).mark_circle(size=150, opacity=0.8, stroke="#ffffff",
                                     strokeWidth=1).encode(
        x=alt.X("dien_tich_m2:Q", title="Diện tích (m²)",
                scale=alt.Scale(zero=False, nice=True)),
        y=alt.Y("gia_ban_ty:Q", title="Giá rao (tỷ đồng)",
                scale=alt.Scale(zero=False, nice=True)),
        color=alt.Color("nhan:N", title="Nhãn giá",
                        scale=alt.Scale(domain=thu_tu,
                                        range=[LUC, XANH, CAM]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("ma_tin:N", title="Mã tin"),
                 alt.Tooltip("tieu_de:N", title="Tiêu đề"),
                 alt.Tooltip("area_key:N", title="Phường"),
                 alt.Tooltip("gia_ban_ty:Q", title="Giá (tỷ)", format=".2f"),
                 alt.Tooltip("dien_tich_m2:Q", title="Diện tích (m²)", format=".0f"),
                 alt.Tooltip("don_gia_trieu_m2:Q", title="Mỗi m² (tr)", format=".0f"),
                 alt.Tooltip("phu_hop_pct:Q", title="Phù hợp (%)", format=".0f"),
                 alt.Tooltip("nhan:N", title="Nhãn giá")])

    moc = pd.DataFrame([{"x": float(yc["dien_tich_m2"]),
                         "y": float(kq["pred_ty"]),
                         "nhan": "Căn bạn mô tả"}])
    thoi = alt.Chart(moc).mark_point(shape="diamond", size=300, filled=True,
                                     color=MUC).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("nhan:N", title="")])
    chu = alt.Chart(moc).mark_text(align="left", dx=14, fontSize=12,
                                   fontWeight="bold", color=MUC).encode(
        x="x:Q", y="y:Q", text="nhan:N")

    st.altair_chart(
        (diem + thoi + chu).add_params(zoom_giu_shift()).properties(height=300),
        use_container_width=True)
    st.caption(
        "Mỗi chấm là một tin gợi ý, màu theo nhãn giá. Dấu thoi đen là căn "
        "bạn mô tả, đặt ở mức giá ứng dụng ước tính. Di chuột lên một chấm để "
        "xem chi tiết, giữ Shift và lăn chuột để phóng to."
    )


def _so(x, dang="{:.0f}") -> str:
    """Số có thể thiếu trong tin đăng, thiếu thì ghi dấu gạch."""
    return "–" if pd.isna(x) else dang.format(x)


def the_tin(r: pd.Series, kq: dict):
    """Một tin đăng trong danh sách gợi ý, bấm vào xem phân tích giá."""
    nhan, mau = NHAN_NGAN[str(r["verdict"])]
    with st.container(border=True):
        c1, c2, c3 = st.columns([3.2, 1.05, 1.35])
        with c1:
            st.markdown(
                f"<div style='font-size:15px;font-weight:600;line-height:1.35'>"
                f"{str(r['tieu_de'])[:92]}</div>"
                f"<div style='font-size:12.5px;color:{XAM};margin-top:4px'>"
                f"{r['ma_tin']} &nbsp;·&nbsp; {r['area_key']} &nbsp;·&nbsp; "
                f"{'Chưa rõ đường' if pd.isna(r['street']) else r['street']} "
                f"&nbsp;·&nbsp; {r['loai_hinh']}</div>"
                f"<div style='font-size:12.5px;color:{XAM};margin-top:2px'>"
                f"{_so(r['dien_tich_m2'])} m² &nbsp;·&nbsp; "
                f"{_so(r['so_phong_ngu'])} phòng ngủ &nbsp;·&nbsp; "
                f"{_so(r['so_phong_ve_sinh'])} vệ sinh &nbsp;·&nbsp; "
                f"{_so(r['tong_so_tang'])} tầng &nbsp;·&nbsp; ngang "
                f"{_so(r['chieu_ngang_m'], '{:.1f}')} m</div>",
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='font-size:19px;font-weight:700;color:{XANH_DAM}'>"
                f"{r['gia_ban_ty']:.2f} tỷ</div>"
                f"<div style='font-size:12.5px;color:{XAM}'>"
                f"{r['don_gia_trieu_m2']:.0f} tr mỗi m²</div>",
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div style='display:inline-block;padding:3px 10px;"
                f"border-radius:11px;background:{mau}1f;color:{mau};"
                f"font-size:12px;font-weight:600'>{nhan}</div>"
                f"<div style='font-size:12.5px;color:{XAM};margin-top:4px'>"
                f"Phù hợp {r['phu_hop'] * 100:.0f}%</div>",
                unsafe_allow_html=True)
            mo = st.session_state.get("mua_chi_tiet") == r["ma_tin"]
            if st.button("Thu gọn" if mo else "Phân tích giá",
                         key=f"ct_{r['ma_tin']}", use_container_width=True):
                st.session_state["mua_chi_tiet"] = None if mo else r["ma_tin"]
                mo = not mo
        if mo:
            phan_tich_gia(r, kq)


def phan_tich_gia(r: pd.Series, kq: dict):
    """Giải thích vì sao một tin đăng được gắn nhãn giá như vậy."""
    ket = str(r["verdict"])
    mau, icon, tom_tat = KET_LUAN[ket]
    st.divider()
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{mau}18;"
        f"border-left:5px solid {mau}'>"
        f"<div style='font-size:19px;font-weight:700;color:{mau}'>"
        f"{icon} {tom_tat}</div></div>",
        unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Người bán rao", f"{r['gia_ban_ty']:.2f} tỷ",
              f"{r['don_gia_trieu_m2']:.0f} tr mỗi m²")
    c2.metric("Mức hợp lý theo ứng dụng", f"{r['pred_ty']:.2f} tỷ")
    lech = (r["gia_ban_ty"] - r["pred_ty"]) / r["pred_ty"] * 100
    c3.metric("Chênh lệch", f"{lech:+.0f}%")

    band_r = bands[bands["area_key"] == r["area_key"]]
    if len(band_r):
        thanh_gia(band_r.iloc[0], float(r["don_gia_trieu_m2"]),
                  f"{r['don_gia_trieu_m2']:.0f} tr/m²")

    if ket == "Giá quá đắt":
        st.warning(
            "**Bạn nên làm gì tiếp theo.** Hỏi người bán vì sao giá cao hơn "
            "mặt bằng khu vực. Có thể căn nhà có điểm đặc biệt chưa thể hiện "
            "trong tin đăng, chẳng hạn vừa sửa lớn, vị trí góc hai mặt tiền, "
            "hoặc đã hoàn công. Nếu không có lý do rõ ràng, hãy so sánh thêm "
            "vài tin khác cùng phường trước khi đặt cọc."
        )
    elif ket == "Giá quá rẻ":
        st.info(
            "**Bạn nên làm gì tiếp theo.** Giá thấp bất thường không hẳn là cơ "
            "hội tốt. Hãy kiểm tra kỹ giấy tờ pháp lý, tình trạng tranh chấp, "
            "quy hoạch, và diện tích thực tế so với diện tích ghi trong tin. "
            "Cũng có khả năng người bán gõ nhầm số."
        )
    else:
        st.success(
            "**Bạn nên làm gì tiếp theo.** Mức giá này phù hợp với mặt bằng "
            "khu vực. Việc còn lại là kiểm tra giấy tờ pháp lý và xem nhà trực "
            "tiếp, hai thứ mà ứng dụng không thể thay bạn làm."
        )

    with st.expander("Vì sao ứng dụng kết luận như vậy?"):
        sig = pd.DataFrame({
            "Dấu hiệu": list(TEN_TIN_HIEU.values()),
            "Mức độ": [float(r[k]) for k in TEN_TIN_HIEU],
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
            f"Ứng dụng xét bốn dấu hiệu độc lập. Tin này có "
            f"**{int(r['n_signals'])} trên 4 dấu hiệu** cho thấy bất thường. "
            "Thanh càng dài nghĩa là dấu hiệu đó càng rõ."
        )
        st.caption(
            f"Ứng dụng ước tính {kq['pred_ty']:.2f} tỷ cho căn bạn mô tả, và "
            f"{r['pred_ty']:.2f} tỷ cho căn trong tin này, tính theo đúng "
            "thông tin người bán đã khai."
        )


# ══════════════════════════════ Xử lý hàng loạt cho quản trị viên
def thanh_hang_loat(cho: pd.DataFrame, chon: list[str], qd: dict):
    """Cho đăng hoặc trả lại nhiều tin cùng lúc, có bước hỏi lại.

    Duyệt từng tin một là đúng cho tin khó, nhưng hàng đợi 397 tin thì cần
    lối đi nhanh. Mọi thao tác hàng loạt đều phải xác nhận và nói rõ số tin,
    vì nhấn nhầm ở đây ảnh hưởng tới hàng trăm người bán.
    """
    xn = st.session_state.get("xn_hang_loat")

    with st.container(border=True):
        st.markdown("**Xử lý hàng loạt**")

        if xn:
            n = len(xn["ma_tin"])
            st.warning(
                f"Bạn sắp **{xn['quyet_dinh'].lower()}** {n} tin "
                f"({xn['nguon']}). Quyết định sẽ ghi cho cả {n} tin trong "
                "phiên làm việc này."
            )
            x1, x2, _ = st.columns([1.2, 1.2, 3.6])
            if x1.button("Đồng ý", type="primary",
                         use_container_width=True):
                for m in xn["ma_tin"]:
                    qd[m] = xn["quyet_dinh"]
                    st.session_state.pop(f"sel_{m}", None)
                st.session_state["xn_hang_loat"] = None
                st.rerun()
            if x2.button("Huỷ", use_container_width=True):
                st.session_state["xn_hang_loat"] = None
                st.rerun()
            return

        def _hoi(quyet_dinh: str, ma_tin: list[str], nguon: str):
            st.session_state["xn_hang_loat"] = {
                "quyet_dinh": quyet_dinh, "ma_tin": list(ma_tin),
                "nguon": nguon}
            st.rerun()

        tat_ca = list(cho["ma_tin"])
        nguon_loc = "toàn bộ tin khớp bộ lọc"
        nguon_chon = "các tin bạn đã tích chọn"

        # Nhãn nút để ngắn, số lượng nói ở dòng tiêu đề phía trên và nói lại
        # một lần nữa ở bước xác nhận, nên không sợ bấm nhầm vì thiếu chữ.
        g_loc, g_chon = st.columns(2)
        with g_loc:
            st.caption(f"**Toàn bộ {len(tat_ca)} tin khớp bộ lọc**, kể cả "
                       "tin chưa hiện trên màn hình")
            b1, b2 = st.columns(2)
            if b1.button("Cho đăng", key="hl_ok_tat_ca",
                         use_container_width=True):
                _hoi("Cho đăng", tat_ca, nguon_loc)
            if b2.button("Trả lại", key="hl_no_tat_ca",
                         use_container_width=True):
                _hoi("Trả lại", tat_ca, nguon_loc)
        with g_chon:
            st.caption(f"**{len(chon)} tin bạn đã tích chọn** trong danh "
                       "sách bên dưới" if chon
                       else "**Chưa tích chọn tin nào** trong danh sách "
                            "bên dưới")
            b3, b4 = st.columns(2)
            if b3.button("Cho đăng", key="hl_ok_chon", disabled=not chon,
                         use_container_width=True):
                _hoi("Cho đăng", chon, nguon_chon)
            if b4.button("Trả lại", key="hl_no_chon", disabled=not chon,
                         use_container_width=True):
                _hoi("Trả lại", chon, nguon_chon)


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
        c1, c2, c3 = st.columns([1, 1, 1.4])
        c1.metric("Giá tham khảo", f"{kq['pred_ty']:.2f} tỷ")
        c2.metric("Mỗi m²", f"{kq['pred_unit_price']:.0f} tr")
        c3.metric("Khoảng hợp lý",
                  f"{kq['low_ty']:.1f} tới {kq['high_ty']:.1f} tỷ")

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

# ══════════════════════════════════════════ 1b. Người mua tìm nhà
elif page == "Tìm nhà phù hợp":
    kho = load_tin_dang()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("Tìm căn nhà hợp với bạn")
        st.write(
            "Mô tả căn nhà bạn muốn mua. Ứng dụng tự ước tính mức giá hợp lý "
            f"cho căn đó, rồi lọc trong **{len(kho):,} tin đăng thật** ra "
            "những căn gần với mô tả nhất. Bạn không cần biết trước giá, và "
            "mỗi tin đều được chấm sẵn xem giá có bất thường hay không."
        )
    with c2:
        st.image(str(FIG / "vai_nguoi_mua.png"), use_container_width=True)

    yc = bieu_mau("mua")

    if st.button("Tìm nhà phù hợp", type="primary", use_container_width=True):
        kq = predict_price(yc)
        st.session_state["mua_tim"] = (yc, kq)
        st.session_state.pop("mua_chi_tiet", None)
        # Xoá bộ lọc của lần tìm trước. Phải xoá chứ không gán đè, vì
        # Streamlit không cho sửa giá trị của một ô điều khiển đã dựng; xoá
        # khoá thì ô đó quay về mặc định tính theo căn nhà vừa mô tả.
        for _k in MAC_DINH_LOC:
            st.session_state.pop(_k, None)

    if "mua_tim" in st.session_state:
        yc_tim, kq = st.session_state["mua_tim"]
        akey = yc_tim["area_key"]
        quan = akey.split(" / ")[0]
        band = bands[bands["area_key"] == akey].iloc[0]

        # Đọc giá trị bộ lọc trước khi vẽ danh sách. Các ô điều khiển nằm
        # bên dưới kết quả, nhưng Streamlit chạy lại cả trang sau mỗi thao
        # tác nên giá trị trong session_state đã là giá trị mới nhất.
        md = dict(MAC_DINH_LOC, loc_tam_gia=tam_gia_mac_dinh(kq["pred_ty"]))
        loc = {k: st.session_state.get(k, v) for k, v in md.items()}
        pham_vi = loc["loc_pham_vi"]
        gia_min, gia_max = loc["loc_tam_gia"]
        pn_min, dt_min = loc["loc_pn"], loc["loc_dt"]
        chi_an_toan, sap_xep = loc["loc_an_toan"], loc["loc_sap_xep"]
        so_tin = loc["loc_so_tin"]

        if pham_vi == "Phường tôi chọn":
            pool = kho[kho["area_key"] == akey]
        elif pham_vi == "Cả quận":
            pool = kho[kho["district"] == quan]
        else:
            pool = kho

        giu = pool["gia_ban_ty"] >= gia_min
        if gia_max < GIA_TRAN:
            giu &= pool["gia_ban_ty"] <= gia_max
        if pn_min:
            giu &= pool["so_phong_ngu"].fillna(0) >= pn_min
        if dt_min:
            giu &= pool["dien_tich_m2"] >= dt_min
        if chi_an_toan:
            giu &= pool["verdict"] == "Bình thường"

        hop = pool[giu].copy()
        if len(hop):
            hop["phu_hop"] = cham_phu_hop(hop, yc_tim, kq["pred_ty"])
            hop = hop.sort_values(THU_TU_SAP_XEP[sap_xep][0],
                                  ascending=THU_TU_SAP_XEP[sap_xep][1])
        ds = hop.head(so_tin)

        st.divider()
        st.subheader("Căn như bạn mô tả đáng khoảng bao nhiêu")
        c1, c2, c3 = st.columns([1, 1, 1.4])
        c1.metric("Mức giá hợp lý", f"{kq['pred_ty']:.2f} tỷ")
        c2.metric("Mỗi m²", f"{kq['pred_unit_price']:.0f} tr")
        c3.metric("Khoảng hợp lý",
                  f"{kq['low_ty']:.1f} tới {kq['high_ty']:.1f} tỷ")
        st.write(
            f"Đây là mức ứng dụng ước tính cho căn bạn vừa mô tả ở **{akey}**, "
            "dùng làm mốc để chọn tin gợi ý bên dưới. Bạn không phải nhập giá, "
            "và cũng có thể tự chỉnh lại tầm giá ở phần tìm kỹ hơn."
        )
        thanh_gia(band, kq["pred_unit_price"],
                  f"{kq['pred_unit_price']:.0f} tr/m²")

        st.divider()
        cach_xep = ("gần nhất với mô tả của bạn" if sap_xep == "Phù hợp nhất"
                    else f"xếp theo {sap_xep.lower()}")
        st.subheader(f"{len(ds)} tin đăng {cach_xep}")
        if not len(ds):
            st.warning(
                "Không có tin nào khớp với bộ lọc hiện tại. Hãy nới tầm giá "
                "hoặc mở rộng phạm vi ở phần **Tìm kỹ hơn** bên dưới."
            )
        else:
            st.caption(
                f"Chọn ra từ {len(hop):,} tin thoả bộ lọc trong "
                f"{len(pool):,} tin của phạm vi đang xem. Độ phù hợp tính theo "
                "vị trí, diện tích, tầm giá, số phòng ngủ, loại nhà và giấy tờ."
            )
            bieu_do_goi_y(ds, kq, yc_tim)

            for _, r in ds.iterrows():
                the_tin(r, kq)

            if len(hop) > len(ds):
                st.caption(
                    f"Còn {len(hop) - len(ds):,} tin khác thoả bộ lọc. Tăng số "
                    "tin hiển thị ở phần Tìm kỹ hơn để xem thêm."
                )

        # ─────────────────────────── Tìm kỹ hơn, đặt sau danh sách gợi ý
        st.divider()
        with st.expander("Tìm kỹ hơn: lọc và sắp xếp lại danh sách", expanded=False):
            st.caption(
                "Mọi thay đổi ở đây áp dụng ngay cho danh sách phía trên."
            )
            c1, c2 = st.columns(2)
            with c1:
                st.radio("Phạm vi tìm", PHAM_VI,
                         index=PHAM_VI.index(md["loc_pham_vi"]),
                         key="loc_pham_vi")
                st.slider("Tầm giá (tỷ đồng)", GIA_SAN, GIA_TRAN,
                          value=md["loc_tam_gia"], step=0.5, key="loc_tam_gia")
                st.caption(
                    f"Kéo mốc trên tới {GIA_TRAN:.0f} tỷ nghĩa là không giới "
                    "hạn giá trần."
                )
            with c2:
                st.slider("Số phòng ngủ tối thiểu", 0, 6,
                          value=md["loc_pn"], key="loc_pn",
                          help="Để 0 nếu bạn không đặt yêu cầu.")
                st.number_input("Diện tích tối thiểu (m²)", 0, 500,
                                value=md["loc_dt"], step=5, key="loc_dt")
                st.checkbox("Chỉ hiện tin có giá bình thường",
                            value=md["loc_an_toan"], key="loc_an_toan")
            c3, c4 = st.columns(2)
            c3.radio("Sắp xếp theo", list(THU_TU_SAP_XEP), key="loc_sap_xep",
                     index=list(THU_TU_SAP_XEP).index(md["loc_sap_xep"]))
            c4.slider("Số tin hiển thị", 5, 30, value=md["loc_so_tin"],
                      key="loc_so_tin")

        st.caption(
            "Nhãn giá là gợi ý để bạn xem xét kỹ hơn, không phải kết luận về "
            "việc người bán có gian dối hay không. Giá lệch mặt bằng có thể do "
            "nhiều nguyên nhân hợp lý."
        )

# ═════════════════════════════════════════ 1c. Quản trị duyệt tin
elif page == "Duyệt tin đăng":
    st.title("Hàng đợi kiểm duyệt")
    st.write(
        "Những tin đăng bị hệ thống gắn cờ vì mức giá bất thường so với khu "
        "vực. Mỗi tin cần một người xem lại rồi quyết định cho đăng hay trả "
        "về cho người bán sửa."
    )

    hd = load_hang_doi()
    # Quyết định giữ trong phiên làm việc. Tải lại trang là mất, đúng như
    # thiết kế: đây là bản minh hoạ luồng kiểm duyệt, chưa nối cơ sở dữ liệu.
    if "quyet_dinh" not in st.session_state:
        st.session_state["quyet_dinh"] = {}
    qd = st.session_state["quyet_dinh"]

    da_duyet = sum(1 for v in qd.values() if v == "Cho đăng")
    da_tu_choi = sum(1 for v in qd.values() if v == "Trả lại")
    con_lai = len(hd) - len(qd)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tin chờ xử lý", f"{con_lai:,}")
    c2.metric("Đã cho đăng", f"{da_duyet:,}")
    c3.metric("Đã trả lại", f"{da_tu_choi:,}")
    c4.metric("Tỷ lệ trả lại",
              f"{da_tu_choi / len(qd) * 100:.0f}%" if qd else "chưa có")
    st.progress(len(qd) / len(hd) if len(hd) else 0.0)
    st.caption(f"Đã xử lý {len(qd)} trên {len(hd)} tin. Quyết định chỉ lưu "
               "trong phiên làm việc này, tải lại trang là bắt đầu lại.")

    st.divider()
    f1, f2, f3 = st.columns([2, 2, 1.4])
    loc_kl = f1.multiselect("Lọc theo kết luận",
                            ["Giá quá đắt", "Giá quá rẻ"],
                            default=["Giá quá đắt", "Giá quá rẻ"])
    loc_quan = f2.multiselect(
        "Lọc theo quận",
        sorted(hd["area_key"].str.split(" / ").str[0].unique()))
    moi_trang = f3.selectbox("Số tin mỗi lần", [5, 10, 20], index=0)

    cho = hd[~hd["ma_tin"].isin(qd) & hd["verdict_B"].isin(loc_kl)]
    if loc_quan:
        cho = cho[cho["area_key"].str.split(" / ").str[0].isin(loc_quan)]

    if cho.empty:
        st.success("Không còn tin nào khớp bộ lọc. Hàng đợi đã sạch.")
    else:
        hien = cho.head(moi_trang)
        # Ô tích chỉ tồn tại cho những thẻ đang hiện, nên lựa chọn cũng chỉ
        # tính trong phạm vi đó. Người duyệt không vô tình xử lý tin mình
        # chưa nhìn thấy.
        chon = [m for m in hien["ma_tin"]
                if st.session_state.get(f"sel_{m}")]
        thanh_hang_loat(cho, chon, qd)

        st.subheader(f"{len(cho)} tin đang chờ, hiện {len(hien)} tin "
                     "có điểm bất thường cao nhất")

        TEN_TH = {"S_resid": "Lệch so với giá ứng dụng ước tính",
                  "S_min_max": "Vượt ngoài khoảng giá từng ghi nhận",
                  "S_percentile": "Vị trí so với các căn cùng phường",
                  "S_ml": "Tổng thể căn nhà khác thường"}

        for _, r in hien.iterrows():
            mau, icon, _ = KET_LUAN[r["verdict_B"]]
            with st.container(border=True):
                a, b = st.columns([5, 2])
                a.markdown(f"**{r['ma_tin']}**  ·  {str(r['tieu_de'])[:95]}")
                b.markdown(
                    f"<div style='text-align:right;color:{mau};font-weight:700'>"
                    f"{icon} {r['verdict_B']}</div>", unsafe_allow_html=True)
                st.caption(
                    f"{r['area_key']}"
                    + (f"  ·  {r['street']}" if isinstance(r["street"], str) else "")
                    + f"  ·  {r['loai_hinh']}")

                def _ty(v):
                    return f"{v:.0f} tỷ" if v >= 100 else f"{v:.2f} tỷ"

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Giá rao", _ty(r["gia_ban_ty"]))
                m2.metric("Ước tính", _ty(r["pred_ty"]))
                lech = (r["gia_ban_ty"] - r["pred_ty"]) / r["pred_ty"] * 100
                m3.metric("Chênh lệch", f"{lech:+.0f}%")
                m4.metric("Điểm", f"{r['score_B']:.0f} / 100",
                          f"{int(r['n_signals_B'])}/4 dấu hiệu",
                          delta_color="off")

                with st.expander("Xem nội dung tin đăng và các dấu hiệu"):
                    st.write(str(r["mo_ta"])[:700] or "Tin đăng không có mô tả.")
                    sig = pd.DataFrame({
                        "Dấu hiệu": [TEN_TH[k] for k in TEN_TH],
                        "Mức độ": [float(r[k]) for k in TEN_TH],
                    })
                    nen_tr = alt.Chart(sig.assign(full=1.0)).mark_bar(
                        cornerRadius=4, color="#e8eef6", size=20).encode(
                        x=alt.X("full:Q", scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%", tickCount=6),
                                title="Mức độ bất thường"),
                        y=alt.Y("Dấu hiệu:N", sort=list(sig["Dấu hiệu"]),
                                title=None,
                                axis=alt.Axis(labelLimit=320,
                                              labelOverlap=False)))
                    cot_tr = alt.Chart(sig).mark_bar(
                        cornerRadius=4, color=XANH, size=20).encode(
                        x=alt.X("Mức độ:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("Dấu hiệu:N", sort=list(sig["Dấu hiệu"]),
                                title=None),
                        tooltip=["Dấu hiệu",
                                 alt.Tooltip("Mức độ:Q", format=".0%")])
                    st.altair_chart((nen_tr + cot_tr).properties(height=160),
                                    use_container_width=True)
                    st.caption(
                        f"Khoảng giá phổ biến của phường: "
                        f"{r['pct_p10']:.0f} tới {r['pct_p90']:.0f} phân vị. "
                        f"Giá rao quy ra {r['don_gia_trieu_m2']:.0f} triệu "
                        "mỗi m²."
                    )

                d0, d1, d2, _ = st.columns([1.1, 1.3, 1.3, 2.3])
                d0.checkbox("Chọn", key=f"sel_{r['ma_tin']}")
                if d1.button("Cho đăng", key=f"ok_{r['ma_tin']}",
                             use_container_width=True):
                    qd[r["ma_tin"]] = "Cho đăng"
                    st.session_state.pop(f"sel_{r['ma_tin']}", None)
                    st.rerun()
                if d2.button("Trả lại", key=f"no_{r['ma_tin']}",
                             type="primary", use_container_width=True):
                    qd[r["ma_tin"]] = "Trả lại"
                    st.session_state.pop(f"sel_{r['ma_tin']}", None)
                    st.rerun()

    if qd:
        st.divider()
        st.subheader("Những tin đã xử lý trong phiên này")
        da = (hd[hd["ma_tin"].isin(qd)]
              .assign(quyet_dinh=lambda x: x["ma_tin"].map(qd))
              [["ma_tin", "quyet_dinh", "verdict_B", "area_key",
                "gia_ban_ty", "pred_ty", "score_B", "tieu_de"]]
              .rename(columns={
                  "ma_tin": "Mã tin", "quyet_dinh": "Quyết định",
                  "verdict_B": "Kết luận máy", "area_key": "Khu vực",
                  "gia_ban_ty": "Giá rao (tỷ)",
                  "pred_ty": "Ước tính (tỷ)", "score_B": "Điểm",
                  "tieu_de": "Tiêu đề"})
              .round(2))
        st.dataframe(da, use_container_width=True, hide_index=True,
                     height=min(400, 60 + 35 * len(da)))
        g1, g2 = st.columns([1.4, 4])
        g1.download_button(
            "Tải CSV",
            da.to_csv(index=False).encode("utf-8-sig"),
            "ket_qua_kiem_duyet.csv", "text/csv",
            use_container_width=True)
        if g2.button("Xoá hết quyết định và làm lại"):
            st.session_state["quyet_dinh"] = {}
            st.session_state["xn_hang_loat"] = None
            for _k in [k for k in st.session_state if k.startswith("sel_")]:
                st.session_state.pop(_k, None)
            st.rerun()

    st.divider()
    st.caption(
        "Hệ thống chỉ gắn cờ để con người xem lại, không tự quyết định. "
        "Giá lệch mặt bằng có thể do nhiều nguyên nhân hợp lý, chẳng hạn căn "
        "nhà có điểm đặc biệt chưa ghi trong tin, hoặc người bán gõ nhầm số. "
        "Quyết định cuối cùng luôn thuộc về quản trị viên."
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
        .properties(height=240).add_params(zoom_giu_shift()),
        use_container_width=True)
    st.caption(
        "Mỗi chấm là một tin đăng. Điều đáng chú ý là **chênh lệch bên trong "
        "cùng một quận còn lớn hơn chênh lệch giữa ba quận**, nên khi so giá "
        "thì nên so ở cấp phường chứ đừng so cấp quận. Giữ Shift rồi cuộn để "
        "phóng to, kéo "
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
            ).properties(height=260).add_params(zoom_giu_shift()), use_container_width=True)
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

Mô tả căn nhà bạn muốn. Ứng dụng ước tính mức giá hợp lý cho căn đó rồi gợi ý
những tin đăng thật gần với mô tả nhất, mỗi tin kèm nhãn cho biết giá có bất
thường hay không. Bấm vào một tin để xem phân tích đầy đủ.
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
