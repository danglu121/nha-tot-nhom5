# GUI Topic 1: Định giá nhà và phát hiện giá bất thường (Nhà Tốt)

**Đồ án tốt nghiệp Data Science, Nhóm 5**
Lư Trần Nguyên Đăng · Nguyễn Anh Duy · Phạm Thị Phương Thúy
*Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*

Ứng dụng Streamlit cho Topic 1: **House-price prediction and Anomaly detection**, dựa
trên 7.935 tin đăng bán nhà tại Bình Thạnh, Gò Vấp và Phú Nhuận.

---

## 1. Ứng dụng dành cho ai

Người dùng chọn vai khi vào app, và các mục sẽ đổi theo vai đó.

| Vai | Mục chính | Nội dung |
|---|---|---|
| **Tôi đang bán nhà** | Định giá nhà của tôi | Điền thông tin căn nhà để nhận giá tham khảo kèm khoảng hợp lý. Nhập thêm mức giá định rao để biết nó nằm ở đâu so với thị trường |
| **Tôi đang tìm mua nhà** | Tìm nhà phù hợp | Mô tả căn nhà muốn mua, không phải nhập giá. Ứng dụng ước tính mức giá hợp lý cho căn đó rồi gợi ý những tin đăng thật gần với mô tả nhất, mỗi tin kèm nhãn giá. Bấm vào một tin để xem phân tích đầy đủ, và lọc kỹ hơn ngay dưới danh sách |
| **Tôi là quản trị viên** | Duyệt tin đăng | Hàng đợi 397 tin bị gắn cờ, xem tiêu đề, nội dung và bốn dấu hiệu rồi quyết định cho đăng hay trả lại người bán. Duyệt từng tin, hoặc xử lý hàng loạt theo bộ lọc và theo nhóm tự tích chọn, luôn có bước xác nhận ghi rõ số tin |

Hai mục còn lại dùng chung cho cả hai vai:

| Mục | Nội dung |
|---|---|
| **Mặt bằng giá khu vực** | Chọn quận để xem khoảng giá từng phường và phường nào đang nhích giá |
| **Về ứng dụng** | Viết cho người thường đọc: số liệu đến từ đâu, chính xác tới đâu, khi nào không nên tin. Phần kỹ thuật đặt trong mục mở rộng cho ai quan tâm |

Toàn bộ nội dung viết theo hướng **giải quyết việc của người dùng**, không trình
bày quy trình xây dựng mô hình.

## 2. Mô hình đứng sau

| Hạng mục | Nội dung |
|---|---|
| Định giá | XGBoost trên 7.935 tin đăng |
| Sai số | MAE 1,22 tỷ · MAPE 14,2% · R² trên thang log 0,877 |
| Phát hiện bất thường | Tổ hợp 4 tín hiệu: phần dư mô hình, vi phạm Min/Max, vị trí phân vị, và Isolation Forest |
| Kết quả chấm toàn bộ | 7.538 bình thường · 245 giá quá rẻ · 152 giá quá đắt |

Bài toán phát hiện bất thường **không có nhãn đúng**. Bốn tín hiệu được tổ hợp bằng
trọng số đặt theo lập luận, không phải học từ dữ liệu. Ứng dụng vì vậy trình bày kết
quả như một **cảnh báo để con người xem xét**, không phải kết luận gian lận.

## 3. Cấu trúc thư mục

```
GUI_Project1/
├── app.py                  ứng dụng Streamlit
├── scoring.py              hàm dự đoán giá và chấm điểm bất thường
├── build_tin_dang.py       dựng lại data/tin_dang.parquet từ dữ liệu gốc
├── models/
│   ├── price_model.joblib      XGBoost định giá
│   ├── isolation_forest.joblib mô hình phát hiện bất thường
│   ├── area_bands.parquet      khung giá tham chiếu theo phường
│   └── metadata.json           tuỳ chọn dropdown, trọng số, ngưỡng
├── data/
│   ├── anomaly_scores.parquet  điểm đã chấm sẵn cho 7.935 tin
│   ├── hang_doi_duyet.parquet  397 tin bị gắn cờ, kèm tiêu đề và mô tả
│   └── tin_dang.parquet        kho tin cho trang tìm nhà của người mua
├── figures/
│   ├── banner.png              ảnh đầu trang, tự dựng bằng matplotlib
│   └── 01..19_*.png            19 biểu đồ phân tích từ notebook
├── requirements.txt
├── setup.sh
├── Procfile
├── runtime.txt
└── .streamlit/config.toml
```

## 4. Dữ liệu được đóng gói ở mức tối thiểu

Đề bài ghi rõ dữ liệu không được chia sẻ, nên repo này **không chứa dữ liệu gốc**. Tập
tin `anomaly_scores.parquet` chỉ giữ 11 cột mà bảng điều khiển thực sự dùng, thay vì
22 cột của bảng gốc, và không kèm tiêu đề hay mô tả tin đăng.

Trang tìm nhà của người mua cần hiển thị tin đăng thật nên `tin_dang.parquet` có giữ
tiêu đề, nhưng chỉ 24 trong 49 cột: những trường mà người mua đọc khi lướt tin, cộng
bốn dấu hiệu đã chấm sẵn. Mô tả chi tiết của tin vẫn không được đóng gói, trừ 397 tin
trong hàng đợi duyệt.

## 5. Chạy tại máy

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Mở `http://localhost:8501`.

## 6. Triển khai lên Streamlit Cloud

1. Đẩy thư mục này lên một repository GitHub.
2. Vào [share.streamlit.io](https://share.streamlit.io), đăng nhập bằng GitHub, chọn
   **New app**.
3. Chọn repository, nhánh, và đặt **Main file path** là `app.py`.
4. Mở **Advanced settings**, chọn **Python 3.12**. Bước này quan trọng vì các phiên bản
   thư viện trong `requirements.txt` được ghim theo Python 3.12.
5. Bấm **Deploy** rồi chờ khoảng 3 tới 5 phút cho lần dựng đầu tiên.

`setup.sh` và `Procfile` có mặt theo yêu cầu của đề bài. Streamlit Cloud không dùng tới
hai tập tin này, chúng dành cho Heroku hoặc các nền tảng tương tự.

### Nếu bước dựng thất bại

Nguyên nhân hay gặp nhất là xung đột phiên bản. Khi đó hãy nới lỏng `requirements.txt`
bằng cách bỏ số phiên bản, giữ lại đúng tên thư viện:

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
pyarrow
altair
```

Lưu ý rằng `price_model.joblib` và `isolation_forest.joblib` được tạo bằng
scikit-learn 1.9 và xgboost 3.4. Nếu Streamlit Cloud cài phiên bản khác hẳn thì có thể
hiện cảnh báo lúc nạp mô hình.

## Phần tương tác

Biểu đồ người dùng nhìn thấy đều **dựng trực tiếp từ dữ liệu lúc chạy**, không phải
ảnh tĩnh, nên rê chuột xem được chi tiết và lọc được theo ý mình.

| Chỗ nào | Tương tác được gì |
|---|---|
| Dải giá từng phường | Rê chuột lên hình thoi xem trung vị, khoảng phổ biến, số tin đăng |
| Khu vực nào đang tăng giá | Đổi quận là bảng xếp hạng đổi theo |
| So sánh ba quận | 4.000 tin đăng dạng chấm, cuộn để phóng to, kéo để di chuyển |
| Tra giá theo con đường | Chọn phường rồi chọn đường trong 363 đường, so mặt bằng đường với cả phường |
| Có bao nhiêu nhà trong tầm tiền | Thanh trượt khoảng giá, số liệu và biểu đồ tự cập nhật |
| Dải giá trong kết quả định giá | Đặt mức giá vừa tính lên khung giá thật của phường |
| Hàng đợi kiểm duyệt | Lọc theo kết luận và quận, bấm duyệt hoặc trả lại, tải kết quả về CSV |

Ảnh tĩnh chỉ còn trong mục **Chi tiết kỹ thuật** ở trang Về ứng dụng, nơi chúng đóng
vai trò bằng chứng về cách chọn mô hình chứ không phải nội dung cho người dùng cuối.

## Hình ảnh trong ứng dụng

| Tập tin | Nguồn |
|---|---|
| `figures/nhatot_logo.png` | Logo Nhà Tốt, đã cắt viền thừa |
| `figures/hero_nhatot.png` | Logo ghép lên dải ảnh dãy phố do nhóm tự vẽ |
| `figures/hero_nha.png` | Dãy phố, nhóm tự vẽ bằng matplotlib |
| `figures/vai_nguoi_ban.png`, `vai_nguoi_mua.png` | Thẻ chọn vai, nhóm tự vẽ |
| `figures/0*.png`, `1*.png` | Biểu đồ từ notebook, dùng trong mục chi tiết kỹ thuật |

Hình vẽ sinh bằng `draw_art.py`, phần ghép logo nằm trong `prep_images.py`.

**Về quyền sở hữu.** Tên và logo Nhà Tốt thuộc quyền sở hữu của Nhà Tốt. Đây là bài
tập của sinh viên, **không liên kết với Nhà Tốt** và không được Nhà Tốt bảo trợ. Logo
được dùng trong phạm vi học tập để minh hoạ cho bộ dữ liệu mà nhà trường cung cấp.
Ứng dụng ghi rõ điều này ngay dưới ảnh đầu trang.
