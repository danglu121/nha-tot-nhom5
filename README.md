# GUI Topic 1: Định giá nhà và phát hiện giá bất thường (Nhà Tốt)

**Đồ án tốt nghiệp Data Science, Nhóm 5**
Lư Trần Nguyên Đăng · Nguyễn Anh Duy · Phạm Thị Phương Thúy
*Trung Tâm Tin Học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*

Ứng dụng Streamlit cho Topic 1: **House-price prediction and Anomaly detection**, dựa
trên 7.935 tin đăng bán nhà tại Bình Thạnh, Gò Vấp và Phú Nhuận.

---

## 1. Ba chức năng

| Trang | Nội dung |
|---|---|
| **Định giá nhà** | Nhập thông tin căn nhà, nhận giá gợi ý kèm khoảng tin cậy tính theo sai số thực đo của từng phân khúc giá |
| **Kiểm tra giá bất thường** | Chấm điểm một tin đăng theo 4 tín hiệu độc lập rồi đưa ra kết luận kèm giải thích từng tín hiệu |
| **Bảng điều khiển** | Phân bố điểm bất thường toàn bộ 7.935 tin, thống kê theo quận, hàng đợi kiểm duyệt, và kế hoạch giám sát sau triển khai |

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

---

> **LƯU Ý PHÁP LÝ.** TẤT CẢ DỮ LIỆU ĐƯỢC CUNG CẤP CHỈ DÀNH CHO VIỆC HỌC TẬP, KHÔNG
> CHIA SẺ, SỬ DỤNG VỚI MỤC ĐÍCH KHÁC. Vì việc chia sẻ, sử dụng cho các mục đích khác
> đều có thể vi phạm Bản quyền và quyền sở hữu trí tuệ, Điều khoản sử dụng của trang
> web, Bảo vệ dữ liệu cá nhân... → vấn đề pháp lý.
