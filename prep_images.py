"""Chuẩn bị ảnh thương hiệu Nhà Tốt cho ứng dụng.

Hai việc: cắt viền thừa quanh logo, và ghép logo lên dải ảnh dãy phố do
draw_art.py vẽ để tạo ảnh đầu trang. Chạy sau draw_art.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

NGUON = Path(os.environ.get("NHATOT_LOGO",
                            "/home/dang.lu/Downloads/Nha tot/LOGO-NHA-TOT-01.png"))
DICH = Path(__file__).parent / "figures"

logo = Image.open(NGUON).convert("RGBA")
a = np.asarray(logo)
# Giữ lại pixel vừa không trong suốt vừa không phải trắng
co_hinh = (a[..., 3] > 8) & ((a[..., :3] < 246).any(axis=2))
hang, cot = np.where(co_hinh.any(axis=1))[0], np.where(co_hinh.any(axis=0))[0]
logo = logo.crop((cot[0], hang[0], cot[-1] + 1, hang[-1] + 1))
logo.thumbnail((560, 560), Image.LANCZOS)
logo.save(DICH / "nhatot_logo.png")
print(f"  nhatot_logo.png  {logo.size[0]}x{logo.size[1]}")

hero = Image.open(DICH / "hero_nha.png").convert("RGBA")
tl = logo.copy()
tl.thumbnail((int(hero.width * 0.17), int(hero.height * 0.26)), Image.LANCZOS)
# Tấm nền trắng mờ phía sau logo, nếu không thì chữ cam chìm vào nền trời
pad = 14
nen = Image.new("RGBA", (tl.width + pad * 2, tl.height + pad * 2),
                (255, 255, 255, 205))
x, y = int(hero.width * 0.028), int(hero.height * 0.10)
hero.alpha_composite(nen, (x - pad, y - pad))
hero.alpha_composite(tl, (x, y))
hero.convert("RGB").save(DICH / "hero_nhatot.png")
print(f"  hero_nhatot.png  {hero.size[0]}x{hero.size[1]}")
