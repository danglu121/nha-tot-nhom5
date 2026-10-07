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
| **Tôi đang tìm mua nhà** | Kiểm tra một tin đăng | Nhập thông tin tin đăng cùng giá người bán đưa ra, nhận kết luận giá có bất thường hay không kèm gợi ý nên kiểm tra thêm điều gì |

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
GUI_Topic1/
├── app.py                  ứng dụng Streamlit
├── scoring.py              hàm dự đoán giá và chấm điểm bất thường
├── models/
│   ├── price_model.joblib      XGBoost định giá
│   ├── isolation_forest.joblib mô hình phát hiện bất thường
│   ├── area_bands.parquet      khung giá tham chiếu theo phường
│   └── metadata.json           tuỳ chọn dropdown, trọng số, ngưỡng
├── data/
│   └── anomaly_scores.parquet  điểm đã chấm sẵn cho bảng điều khiển
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

## Minh hoạ trong ứng dụng

Ảnh đầu trang và các thẻ chọn vai (nhà phố, biển rao bán, kính lúp soi giá) đều do nhóm **tự vẽ bằng matplotlib**,
mã nguồn nằm trong `draw_art.py`. Không dùng ảnh hay logo của bên thứ ba, nên
không vướng vấn đề bản quyền khi công khai repository.

---

> **LƯU Ý PHÁP LÝ.** TẤT CẢ DỮ LIỆU ĐƯỢC CUNG CẤP CHỈ DÀNH CHO VIỆC HỌC TẬP, KHÔNG
> CHIA SẺ, SỬ DỤNG VỚI MỤC ĐÍCH KHÁC. Vì việc chia sẻ, sử dụng cho các mục đích khác
> đều có thể vi phạm Bản quyền và quyền sở hữu trí tuệ, Điều khoản sử dụng của trang
> web, Bảo vệ dữ liệu cá nhân... → vấn đề pháp lý.
