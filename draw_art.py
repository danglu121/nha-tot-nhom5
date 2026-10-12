"""Vẽ minh hoạ chủ đề cho hai ứng dụng.

Tự vẽ bằng matplotlib thay vì tải ảnh có sẵn, vừa tránh vấn đề bản quyền
vừa giữ đúng bảng màu của ứng dụng.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

mpl.use("Agg")

BASE = Path("/home/dang.lu/Personal/Final ML")
T1 = BASE / "GUI_Project1" / "figures"
P2 = BASE / "GUI_Project2" / "figures"
for d in (T1, P2):
    d.mkdir(parents=True, exist_ok=True)

XANH = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAM, LUC, VANG = "#eb6834", "#1baf7a", "#f5c24c"
TRANG, MUC, XAM = "#ffffff", "#0b0b0b", "#52514e"


def khung(w, h, nen):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(nen)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100 * w / h)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def luu(fig, path, nen):
    fig.savefig(path, facecolor=nen)
    plt.close(fig)
    print(f"  {path.relative_to(BASE)}  {path.stat().st_size / 1024:.0f} KB")


# ═══════════════════════════════════════════════ Chi tiết dùng lại
def cua_so(ax, x, y, w, h, mau=VANG, vien="#ffffff"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=mau, edgecolor=vien,
                           linewidth=1.1, zorder=5))


def nha_pho(ax, x, nen_y, rong, cao, than, mai, n_tang=2, cua_chinh=True):
    """Một căn nhà phố: thân hình chữ nhật, mái dốc, cửa sổ xếp lưới."""
    ax.add_patch(Rectangle((x, nen_y), rong, cao, facecolor=than,
                           edgecolor="none", zorder=3))
    ax.add_patch(Polygon([[x - rong * 0.09, nen_y + cao],
                          [x + rong / 2, nen_y + cao + rong * 0.38],
                          [x + rong * 1.09, nen_y + cao]],
                         closed=True, facecolor=mai, edgecolor="none", zorder=4))
    cs_w, cs_h = rong * 0.21, cao * 0.13
    for t in range(n_tang):
        y = nen_y + cao * (0.74 - 0.27 * t)
        for c in range(2):
            cua_so(ax, x + rong * (0.21 + 0.42 * c), y, cs_w, cs_h)
    if cua_chinh:
        ax.add_patch(Rectangle((x + rong * 0.38, nen_y), rong * 0.24, cao * 0.22,
                               facecolor="#8a5a3b", edgecolor="none", zorder=5))


def toa_nha(ax, x, nen_y, rong, cao, than, n_cot=4, n_hang=7):
    """Toà nhà văn phòng: khối chữ nhật với lưới cửa kính."""
    ax.add_patch(Rectangle((x, nen_y), rong, cao, facecolor=than,
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - rong * 0.04, nen_y + cao), rong * 1.08,
                           cao * 0.035, facecolor="#ffffff", alpha=0.25,
                           edgecolor="none", zorder=4))
    kw = rong * 0.72 / n_cot
    kh = cao * 0.72 / n_hang
    rng = np.random.default_rng(int(x * 7) + n_hang)
    for i in range(n_cot):
        for j in range(n_hang):
            sang = rng.random() < 0.34
            cua_so(ax,
                   x + rong * 0.14 + i * kw + kw * 0.14,
                   nen_y + cao * 0.12 + j * kh + kh * 0.16,
                   kw * 0.72, kh * 0.62,
                   mau=VANG if sang else "#bcd6f2", vien="none")


def cay(ax, x, nen_y, cao, mau=LUC):
    ax.add_patch(Rectangle((x - cao * 0.045, nen_y), cao * 0.09, cao * 0.34,
                           facecolor="#7a5a3f", edgecolor="none", zorder=4))
    for dy, r in [(0.46, 0.21), (0.62, 0.17), (0.76, 0.12)]:
        ax.add_patch(Circle((x, nen_y + cao * dy), cao * r, facecolor=mau,
                            edgecolor="none", zorder=5))


def may(ax, x, y, s, mau="#ffffff", alpha=0.7):
    for dx, dy, r in [(0, 0, 1.0), (0.85, 0.12, 0.74), (-0.8, 0.08, 0.66),
                      (0.3, 0.42, 0.62)]:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, facecolor=mau,
                            alpha=alpha, edgecolor="none", zorder=2))


# ═══════════════════════════════════════ Ảnh đầu trang Topic 1
def hero_nha():
    nen = "#dceaf9"
    fig, ax = khung(16, 4.2, nen)
    W = ax.get_xlim()[1]

    # Mặt trời và mây đặt ở khoảng trời còn trống phía trên nóc nhà cao nhất.
    ax.add_patch(Circle((205, 92), 7.5, facecolor="#ffd05e",
                        edgecolor="none", zorder=1))
    may(ax, 60, 92, 5.5, alpha=0.85)
    may(ax, 262, 93, 4.6, alpha=0.85)
    may(ax, 340, 95, 4.0, alpha=0.85)

    ax.add_patch(Rectangle((0, 0), W, 26, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((0, 0), W, 13, facecolor="#9aa4a0",
                           edgecolor="none", zorder=2))
    for i in range(16):
        ax.add_patch(Rectangle((8 + i * 24, 6), 11, 1.6, facecolor="#e6e9e8",
                               edgecolor="none", zorder=3))

    # Dãy nhà phố cao thấp xen kẽ, màu chuyển dần cho có chiều sâu
    # Chiều cao đã hạ để đỉnh mái dốc cao nhất vẫn nằm dưới mép trên khung.
    cau_hinh = [
        (18, 40, 39, XANH[2], XANH[5], 2), (62, 34, 32, XANH[1], XANH[4], 2),
        (100, 44, 45, XANH[3], XANH[6], 3), (148, 36, 35, XANH[2], XANH[5], 2),
        (188, 42, 42, XANH[1], XANH[4], 2), (234, 33, 30, XANH[3], XANH[6], 2),
        (271, 39, 37, XANH[2], XANH[5], 2), (314, 45, 46, XANH[1], XANH[4], 3),
    ]
    for x, rong, cao, than, mai, nt in cau_hinh:
        nha_pho(ax, x, 26, rong, cao, than, mai, nt)

    for x in (8, 93, 182, 262, 352):
        cay(ax, x, 26, 30)

    luu(fig, T1 / "hero_nha.png", nen)


# ═══════════════════════════════════ Ảnh đầu trang Project 2
def hero_vanphong():
    nen = "#dfe8f6"
    fig, ax = khung(16, 4.2, nen)
    W = ax.get_xlim()[1]

    may(ax, 42, 93, 5.2, alpha=0.8)
    may(ax, 190, 95, 4.4, alpha=0.8)
    may(ax, 288, 94, 4.8, alpha=0.8)

    ax.add_patch(Rectangle((0, 0), W, 17, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))

    # Chiều cao đã hạ để nóc toà cao nhất vẫn nằm dưới mép trên khung.
    thap = [
        (14, 34, 39, XANH[2], 3, 6), (54, 28, 55, XANH[4], 3, 8),
        (88, 40, 45, XANH[1], 4, 7), (134, 30, 62, XANH[5], 3, 9),
        (170, 36, 36, XANH[2], 4, 5), (212, 32, 56, XANH[3], 3, 8),
        (250, 42, 48, XANH[1], 4, 7), (298, 28, 67, XANH[6], 3, 10),
        (332, 36, 42, XANH[2], 4, 6),
    ]
    for x, rong, cao, than, nc, nh in thap:
        toa_nha(ax, x, 17, rong, cao, than, nc, nh)

    for x in (6, 128, 204, 290, 378):
        cay(ax, x, 17, 24)

    luu(fig, P2 / "hero_vanphong.png", nen)


# ═══════════════════════════════════════════ Thẻ chọn vai trò
def the(path, ve, nen="#f2f6fc"):
    fig, ax = khung(4.4, 3.0, nen)
    ve(ax, ax.get_xlim()[1])
    luu(fig, path, nen)


def ve_nguoi_ban(ax, W):
    """Nhà kèm bảng rao bán."""
    nha_pho(ax, W * 0.17, 16, W * 0.44, 46, XANH[2], XANH[5], 2)
    ax.add_patch(Rectangle((0, 0), W, 16, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    # Bảng cắm trước nhà
    ax.add_patch(Rectangle((W * 0.70, 16), 2.4, 34, facecolor="#8a5a3b",
                           edgecolor="none", zorder=6))
    ax.add_patch(FancyBboxPatch((W * 0.60, 48), W * 0.32, 22,
                                boxstyle="round,pad=1.4,rounding_size=3",
                                facecolor=CAM, edgecolor="none", zorder=7))
    ax.text(W * 0.60 + W * 0.16, 59, "BÁN", fontsize=15, fontweight="bold",
            color=TRANG, ha="center", va="center", zorder=8)
    cay(ax, W * 0.07, 16, 26)


def ve_nguoi_mua(ax, W):
    """Nhà kèm kính lúp soi giá."""
    nha_pho(ax, W * 0.10, 16, W * 0.44, 46, XANH[2], XANH[5], 2)
    ax.add_patch(Rectangle((0, 0), W, 16, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    cx, cy, r = W * 0.70, 52, 15
    ax.add_patch(Circle((cx, cy), r, facecolor="#ffffff", alpha=0.92,
                        edgecolor=XANH[6], linewidth=3.4, zorder=7))
    ax.plot([cx + r * 0.72, cx + r * 1.65], [cy - r * 0.72, cy - r * 1.65],
            color=XANH[6], linewidth=5.4, solid_capstyle="round", zorder=7)
    ax.text(cx, cy, "₫", fontsize=20, fontweight="bold", color=CAM,
            ha="center", va="center", zorder=8)
    cay(ax, W * 0.90, 16, 24)


def ve_ung_vien(ax, W):
    """Cặp tài liệu kèm kính lúp, gợi ý việc tìm nơi làm việc."""
    ax.add_patch(Rectangle((0, 0), W, 14, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))
    bx, by, bw, bh = W * 0.16, 22, W * 0.44, 34
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0,rounding_size=3",
                                facecolor=XANH[4], edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((bx + bw * 0.34, by + bh), bw * 0.32, 7,
                           facecolor=XANH[6], edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((bx, by + bh * 0.42), bw, 5, facecolor=XANH[6],
                           edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((bx + bw * 0.43, by + bh * 0.34), bw * 0.14, 13,
                           facecolor=VANG, edgecolor="none", zorder=6))
    cx, cy, r = W * 0.73, 54, 13
    ax.add_patch(Circle((cx, cy), r, facecolor="#ffffff", alpha=0.94,
                        edgecolor=CAM, linewidth=3.2, zorder=7))
    ax.plot([cx + r * 0.72, cx + r * 1.7], [cy - r * 0.72, cy - r * 1.7],
            color=CAM, linewidth=5.0, solid_capstyle="round", zorder=7)


def ve_nha_tuyen_dung(ax, W):
    """Toà nhà văn phòng kèm hai nhân viên."""
    ax.add_patch(Rectangle((0, 0), W, 14, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))
    toa_nha(ax, W * 0.14, 14, W * 0.40, 62, XANH[4], 3, 6)
    toa_nha(ax, W * 0.58, 14, W * 0.26, 44, XANH[2], 2, 4)
    for dx, mau in [(0.08, CAM), (0.93, LUC)]:
        x = W * dx
        ax.add_patch(Circle((x, 28), 5.2, facecolor=mau, edgecolor="none",
                            zorder=8))
        ax.add_patch(FancyBboxPatch((x - 5.4, 14), 10.8, 12,
                                    boxstyle="round,pad=0,rounding_size=3",
                                    facecolor=mau, edgecolor="none", zorder=8))


# ═══════════════════════════════════════════════ Chạy
print("Topic 1:")
hero_nha()
the(T1 / "vai_nguoi_ban.png", ve_nguoi_ban)
the(T1 / "vai_nguoi_mua.png", ve_nguoi_mua)
print("Project 2:")
hero_vanphong()
the(P2 / "vai_ung_vien.png", ve_ung_vien)
the(P2 / "vai_nha_tuyen_dung.png", ve_nha_tuyen_dung)

# ═══════════════════════════════════ Biểu tượng cho từng mục
# Mỗi mục trong ứng dụng có một màu riêng, biểu tượng vẽ theo đúng màu đó để
# người dùng nhận ra mình đang ở đâu mà không phải đọc tiêu đề.
def icon(path, ve, mau_nen):
    fig, ax = khung(3.0, 3.0, mau_nen)
    ve(ax, ax.get_xlim()[1], 100)
    luu(fig, path, mau_nen)


def bang_gia(ax, x, y, w, h, mau, chu):
    """Thẻ giá bo tròn có lỗ treo, dùng cho biểu tượng định giá."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=3",
                                facecolor=mau, edgecolor="none", zorder=8))
    ax.add_patch(Circle((x + w * 0.16, y + h * 0.5), min(w, h) * 0.11,
                        facecolor=TRANG, edgecolor="none", zorder=9))
    ax.text(x + w * 0.60, y + h * 0.5, chu, fontsize=15, fontweight="bold",
            color=TRANG, ha="center", va="center", zorder=9)


def kinh_lup(ax, cx, cy, r, mau, chu=None):
    ax.add_patch(Circle((cx, cy), r, facecolor=TRANG, alpha=0.94,
                        edgecolor=mau, linewidth=3.6, zorder=8))
    ax.plot([cx + r * 0.72, cx + r * 1.62], [cy - r * 0.72, cy - r * 1.62],
            color=mau, linewidth=5.6, solid_capstyle="round", zorder=8)
    if chu:
        ax.text(cx, cy, chu, fontsize=16, fontweight="bold", color=mau,
                ha="center", va="center", zorder=9)


def cot(ax, x, nen_y, w, h, mau):
    ax.add_patch(FancyBboxPatch((x, nen_y), w, h,
                                boxstyle="round,pad=0,rounding_size=2",
                                facecolor=mau, edgecolor="none", zorder=5))


def ve_icon_dinh_gia(ax, W, H):
    """Nhà kèm thẻ giá: người bán muốn biết nhà mình đáng bao nhiêu."""
    ax.add_patch(Rectangle((0, 0), W, 18, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    nha_pho(ax, W * 0.12, 18, W * 0.50, 42, XANH[2], XANH[5], 2)
    bang_gia(ax, W * 0.46, 62, W * 0.46, 20, LUC, "₫")
    cay(ax, W * 0.86, 18, 26)


def ve_icon_tim_nha(ax, W, H):
    """Ba căn nhà và một kính lúp: người mua đang dò tìm."""
    ax.add_patch(Rectangle((0, 0), W, 18, facecolor="#cfe0d2",
                           edgecolor="none", zorder=1))
    for x, rong, cao, than, mai in [
        (W * 0.04, W * 0.26, 30, XANH[1], XANH[4]),
        (W * 0.36, W * 0.28, 40, XANH[3], XANH[6]),
        (W * 0.70, W * 0.24, 26, XANH[2], XANH[5]),
    ]:
        nha_pho(ax, x, 18, rong, cao, than, mai, 2, cua_chinh=False)
    kinh_lup(ax, W * 0.70, 70, 16, "#2a78d6")


def ve_icon_duyet_tin(ax, W, H):
    """Bảng kẹp hồ sơ với một dấu tích và một dấu gạch: duyệt hay trả lại."""
    bx, by, bw, bh = W * 0.18, 14, W * 0.64, 72
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor=TRANG, edgecolor="#c9d4e4",
                                linewidth=2.2, zorder=4))
    ax.add_patch(FancyBboxPatch((bx + bw * 0.30, by + bh - 6), bw * 0.40, 11,
                                boxstyle="round,pad=0,rounding_size=3",
                                facecolor="#7c4dcc", edgecolor="none", zorder=6))
    for i, (mau, tick) in enumerate([(LUC, True), (CAM, False), (LUC, True)]):
        y = by + bh * (0.60 - 0.22 * i)
        ax.add_patch(Rectangle((bx + bw * 0.30, y), bw * 0.48, 4.6,
                               facecolor="#dfe6f1", edgecolor="none", zorder=5))
        cx = bx + bw * 0.17
        ax.add_patch(Circle((cx, y + 2.3), 5.4, facecolor=mau,
                            edgecolor="none", zorder=6))
        if tick:
            ax.plot([cx - 2.4, cx - 0.6, cx + 2.6],
                    [y + 2.3, y + 0.3, y + 4.6], color=TRANG, linewidth=2.2,
                    solid_capstyle="round", zorder=7)
        else:
            ax.plot([cx - 2.4, cx + 2.4], [y + 2.3, y + 2.3], color=TRANG,
                    linewidth=2.2, solid_capstyle="round", zorder=7)


def ve_icon_mat_bang_gia(ax, W, H):
    """Cột giá cao dần, trên mỗi cột là một mái nhà nhỏ."""
    ax.add_patch(Rectangle((0, 12), W, 2.4, facecolor="#c9d4e4",
                           edgecolor="none", zorder=2))
    cao = [26, 40, 33, 54, 46]
    mau = [XANH[1], XANH[2], XANH[3], "#d99100", XANH[5]]
    bw = W * 0.14
    for i, (h, m) in enumerate(zip(cao, mau)):
        x = W * 0.07 + i * (bw + W * 0.035)
        cot(ax, x, 14, bw, h, m)
        ax.add_patch(Polygon([[x - bw * 0.12, 14 + h],
                              [x + bw * 0.5, 14 + h + bw * 0.42],
                              [x + bw * 1.12, 14 + h]],
                             closed=True, facecolor=m, edgecolor="none",
                             zorder=6))


def ve_icon_ve_ung_dung(ax, W, H, mau="#46617d"):
    """Quyển sách mở: phần giới thiệu và giải thích."""
    ax.add_patch(Polygon([[W * 0.08, 26], [W * 0.48, 34], [W * 0.48, 76],
                          [W * 0.08, 66]], closed=True, facecolor=mau,
                         edgecolor="none", zorder=4))
    ax.add_patch(Polygon([[W * 0.92, 26], [W * 0.52, 34], [W * 0.52, 76],
                          [W * 0.92, 66]], closed=True, facecolor=mau,
                         alpha=0.74, edgecolor="none", zorder=4))
    for i in range(4):
        y = 44 + i * 8
        ax.plot([W * 0.14, W * 0.42], [y, y + 5], color=TRANG, alpha=0.75,
                linewidth=2.0, solid_capstyle="round", zorder=5)
        ax.plot([W * 0.58, W * 0.86], [y + 5, y], color=TRANG, alpha=0.75,
                linewidth=2.0, solid_capstyle="round", zorder=5)


def ve_icon_tim_cty(ax, W, H):
    """Hai toà nhà và kính lúp: ứng viên đang tìm nơi làm việc."""
    ax.add_patch(Rectangle((0, 0), W, 14, facecolor="#c3cedd",
                           edgecolor="none", zorder=1))
    toa_nha(ax, W * 0.08, 14, W * 0.30, 52, XANH[4], 3, 7)
    toa_nha(ax, W * 0.44, 14, W * 0.24, 36, XANH[2], 2, 5)
    kinh_lup(ax, W * 0.74, 62, 16, CAM)


def ve_icon_doc_danh_gia(ax, W, H):
    """Bong bóng lời nói kèm dải sao: một đánh giá của nhân viên."""
    bx, by, bw, bh = W * 0.10, 34, W * 0.80, 44
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor=XANH[1], edgecolor="none", zorder=4))
    ax.add_patch(Polygon([[bx + bw * 0.22, by], [bx + bw * 0.34, by],
                          [bx + bw * 0.22, by - 12]], closed=True,
                         facecolor=XANH[1], edgecolor="none", zorder=4))
    for i in range(4):
        goc = np.linspace(0, 2 * np.pi, 11)[:-1] + np.pi / 2
        bk = np.where(np.arange(10) % 2 == 0, 5.6, 2.4)
        cx = bx + bw * (0.17 + 0.22 * i)
        mau = VANG if i < 3 else "#dbe4f0"
        ax.add_patch(Polygon(np.c_[cx + bk * np.cos(goc),
                                   by + bh * 0.58 + bk * np.sin(goc)],
                             closed=True, facecolor=mau, edgecolor="none",
                             zorder=6))
    ax.add_patch(Rectangle((bx + bw * 0.14, by + bh * 0.22), bw * 0.72, 4.2,
                           facecolor=TRANG, alpha=0.8, edgecolor="none",
                           zorder=6))


def ve_icon_thi_truong(ax, W, H):
    """Cột thống kê dâng dần trước dãy cao ốc mờ."""
    for x, rong, cao in [(W * 0.06, W * 0.18, 56), (W * 0.40, W * 0.16, 70),
                         (W * 0.70, W * 0.20, 48)]:
        ax.add_patch(Rectangle((x, 14), rong, cao, facecolor=XANH[0],
                               alpha=0.75, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((0, 12), W, 2.4, facecolor="#c9d4e4",
                           edgecolor="none", zorder=3))
    for i, (h, m) in enumerate(zip([22, 38, 30, 52],
                                   [XANH[2], XANH[4], CAM, XANH[6]])):
        cot(ax, W * 0.10 + i * W * 0.21, 14, W * 0.14, h, m)


# ═══════════════════════════════════════════════ Chạy phần biểu tượng
print("Biểu tượng Topic 1:")
icon(T1 / "icon_dinh_gia.png", ve_icon_dinh_gia, "#e7f4ee")
icon(T1 / "icon_tim_nha.png", ve_icon_tim_nha, "#e4eefb")
icon(T1 / "icon_duyet_tin.png", ve_icon_duyet_tin, "#efe9fb")
icon(T1 / "icon_mat_bang_gia.png", ve_icon_mat_bang_gia, "#fbf1dd")
icon(T1 / "icon_ve_ung_dung.png", ve_icon_ve_ung_dung, "#e9eef4")
print("Biểu tượng Project 2:")
icon(P2 / "icon_tim_cty.png", ve_icon_tim_cty, "#e4eefb")
icon(P2 / "icon_doc_danh_gia.png", ve_icon_doc_danh_gia, "#fdeae3")
icon(P2 / "icon_thi_truong.png", ve_icon_thi_truong, "#e7f4ee")
icon(P2 / "icon_ve_ung_dung.png", ve_icon_ve_ung_dung, "#e9eef4")
