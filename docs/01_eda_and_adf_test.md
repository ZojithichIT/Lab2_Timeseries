# Bài 1: Khám Phá Dữ Liệu (EDA) & Nhận Diện Tính Dừng

Tài liệu hướng dẫn và mã nguồn thực hành cho **Bài 1: Khám phá dữ liệu (Exploratory Data Analysis - EDA)**. Toàn bộ mã nguồn được phân tách rõ ràng thành từng cell riêng biệt để tiện copy hoặc chạy trong Jupyter Notebook / Google Colab.

---

## 1. Import Các Thư Viện Cần Thiết

```python
# [Cell 1]: Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Thư viện thống kê chuyên dụng cho Time Series (ADF test)
from statsmodels.tsa.stattools import adfuller

# Thiết lập giao diện biểu đồ
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
```

---

## 2. Các Bước Thực Hiện (Từng Cell Riêng Biệt)

### Bước 1: Load dữ liệu (Riêng)

> Đọc dữ liệu từ file CSV, kiểm tra cấu trúc dữ liệu, các dòng đầu tiên và thông tin kiểu dữ liệu ban đầu.

```python
# [Cell 2]: Load dữ liệu
# Đường dẫn tương đối từ thư mục scripts/ hoặc project root
data_path = '../data/raw/a10.csv'  # Hoặc 'data/raw/a10.csv' nếu chạy từ thư mục gốc

df = pd.read_csv(data_path)

# Hiển thị thông tin tổng quan
print("=== 5 DÒNG ĐẦU TIÊN CỦA DỮ LIỆU ===")
display(df.head())

print("\n=== THÔNG TIN KIỂU DỮ LIỆU BAN ĐẦU ===")
df.info()

print("\n=== THỐNG KÊ MÔ TẢ ===")
display(df.describe())
```

---

### Bước 2: Chuyển `date` thành index thời gian (Riêng)

> Chuyển cột `date` từ kiểu chuỗi ký tự (`object`) sang định dạng chuẩn `datetime`, sau đó thiết lập làm chỉ mục thời gian (`DatetimeIndex`) và gán tần số thời gian (Frequency = 'MS' - Month Start).

```python
# [Cell 3]: Chuyển cột date thành DatetimeIndex
# 1. Chuyển đổi sang định dạng datetime chuẩn
df['date'] = pd.to_datetime(df['date'])

# 2. Đặt cột date làm index của DataFrame
df.set_index('date', inplace=True)

# 3. Thiết lập tần số chu kỳ theo tháng (Month Start)
df = df.asfreq('MS')

# Kiểm tra lại cấu trúc index sau khi chuyển đổi
print("=== KIỂM TRA INDEX VÀ CẤU TRÚC SAU KHI CHUYỂN DATE THÀNH INDEX ===")
print("Kiểu của Index:", type(df.index))
print("Tần số (Frequency):", df.index.freq)
print("Khoảng thời gian khảo sát: Từ", df.index.min().strftime('%Y-%m'), "đến", df.index.max().strftime('%Y-%m'))
print(f"Tổng số quan sát (tháng): {len(df)}")
display(df.head())

# (Tùy chọn) Lưu dữ liệu đã xử lý index vào thư mục data/processed/
import os
os.makedirs('../data/processed', exist_ok=True)
df.to_csv('../data/processed/a10_processed.csv', index=True)
print("\n-> Đã lưu dữ liệu vào: data/processed/a10_processed.csv")
```

---

### Bước 3: Vẽ đồ thị chuỗi thời gian (Riêng)

> Trực quan hóa toàn bộ chuỗi thời gian để quan sát sự biến thiên của dữ liệu theo thời gian, phát hiện xu hướng và tính mùa vụ.

```python
# [Cell 4]: Vẽ đồ thị chuỗi thời gian (Time Series Plot)
plt.figure(figsize=(14, 6))

plt.plot(df.index, df['value'], color='#1f77b4', linewidth=2, label='Doanh số bán thuốc hàng tháng (a10)')

# Trang trí đồ thị
plt.title('Đồ Thị Chuỗi Thời Gian Doanh Số Bán Hàng (Monthly Sales Data - a10)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Thời gian (Năm - Tháng)', fontsize=12, labelpad=10)
plt.ylabel('Doanh số (Triệu USD)', fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Hiển thị đồ thị
plt.show()
```

---

## 3. Câu Hỏi Cần Trả Lời & Phân Tích Chuyên Sâu

Dựa trên việc quan sát đồ thị chuỗi thời gian vừa vẽ ở **Bước 3**, ta đưa ra câu trả lời cho 3 câu hỏi chính:

| Câu hỏi | Kết luận | Giải thích chi tiết từ đồ thị |
| :--- | :---: | :--- |
| **1. Chuỗi có xu hướng (Trend) không?** | **CÓ** | Đường dữ liệu có **xu hướng tăng mạnh và nhất quán (strong upward trend)** theo thời gian từ năm 1991 đến 2008. Giá trị doanh số trung bình tăng dần qua các năm, không dao động quanh một mức trung bình cố định. |
| **2. Có dấu hiệu mùa vụ (Seasonality) không?** | **CÓ** | Đồ thị xuất hiện **các đỉnh (peaks) và đáy (troughs) lặp đi lặp lại rất đều đặn theo chu kỳ hàng năm (chu kỳ 12 tháng - $s=12$)**. Cụ thể, doanh số thường đạt đỉnh cao nhất vào khoảng tháng 12 hoặc tháng 1 hàng năm và giảm sâu xuống mức thấp nhất vào tháng 2 sau đó. Biên độ dao động mùa vụ có xu hướng mở rộng dần theo thời gian (Multiplicative Seasonality). |
| **3. Chuỗi có khả năng dừng (Stationary) không?** | **KHÔNG** | Chuỗi **không có tính dừng (Non-stationary)** vì vi phạm cả hai điều kiện cốt lõi của tính dừng:<br>1. **Kỳ vọng/Trung bình (Mean) không cố định**: Giá trị trung bình tăng dần theo thời gian do có xu hướng tăng.<br>2. **Phương sai (Variance) không cố định**: Biên độ dao động các đỉnh/đáy mùa vụ ngày càng giãn rộng về các năm sau.<br>*(Do đó, chuỗi cần phải qua các bước sai phân $d$ và $D$ ở các bài tiếp theo để đưa về tính dừng trước khi huấn luyện mô hình ARIMA/SARIMA).* |

---

## 4. (Tùy chọn) Mã Nguồn Tương Tự Cho Dataset `airline-passengers.csv`

Nếu bạn thực hành trên tập dữ liệu hàng không **AirPassengers**:

```python
# [Cell Tùy chọn]: Áp dụng cho dataset airline-passengers.csv
# 1. Load dữ liệu
df_air = pd.read_csv('../data/raw/airline-passengers.csv')

# 2. Chuyển cột 'Month' thành datetime index
df_air['Month'] = pd.to_datetime(df_air['Month'])
df_air.set_index('Month', inplace=True)
df_air = df_air.asfreq('MS')

# 3. Vẽ đồ thị
plt.figure(figsize=(14, 6))
plt.plot(df_air.index, df_air['Passengers'], color='#e74c3c', linewidth=2, label='Số lượng hành khách (AirPassengers)')
plt.title('Đồ Thị Chuỗi Thời Gian Số Lượng Hành Khách Hàng Không (1949 - 1960)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Thời gian (Năm - Tháng)', fontsize=12)
plt.ylabel('Hành khách (nghìn người)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

---

# Bài 2: Kiểm Định ADF Trên Chuỗi Gốc (Stationarity Test)

Kiểm định **Augmented Dickey-Fuller (ADF)** là phương pháp thống kê định lượng chuẩn mực nhằm kiểm tra xem một chuỗi thời gian có tính dừng (Stationary) hay không thông qua việc kiểm tra sự tồn tại của nghiệm đơn vị (Unit Root).

---

## 1. Mã Nguồn Kiểm Định ADF (Từng Cell Riêng Biệt)

### Cell: Định nghĩa hàm kiểm định ADF chuẩn hóa

> Định nghĩa hàm `adf_test()` hiển thị trực quan các thông số: ADF Statistic, p-value, số độ trễ đã dùng, số quan sát, các giá trị tới hạn (Critical Values) và đưa ra kết luận tự động theo ngưỡng $\alpha = 0.05$. Hàm này sẽ được tái sử dụng xuyên suốt toàn bộ bài Lab.

```python
# [Cell 5]: Định nghĩa hàm kiểm định ADF
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name='Chuỗi'):
    """
    Thực hiện kiểm định Augmented Dickey-Fuller (ADF) và in kết quả chi tiết
    """
    series_clean = series.dropna()
    result = adfuller(series_clean, autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    used_lag = result[2]
    n_obs = result[3]
    critical_values = result[4]
    
    print(f"==================================================")
    print(f"  KẾT QUẢ KIỂM ĐỊNH ADF CHO: {name.upper()}")
    print(f"==================================================")
    print(f"• ADF Statistic        : {adf_stat:.6f}")
    print(f"• p-value              : {p_value:.6f}")
    print(f"• Số độ trễ đã dùng (Lags): {used_lag}")
    print(f"• Số lượng quan sát    : {n_obs}")
    print(f"• Các giá trị tới hạn (Critical Values):")
    for level, value in critical_values.items():
        print(f"    - Mức {level:>3}: {value:.6f}")
    print(f"--------------------------------------------------")
    
    # Kết luận định lượng dựa trên mức ý nghĩa alpha = 0.05 (5%)
    alpha = 0.05
    if p_value <= alpha:
        print(f" Kết luận: p-value <= {alpha} => BÁC BỎ H0.")
        print(f"==> Chuỗi '{name}' LÀ CHUỖI DỪNG (Stationary).")
        is_stationary = True
    else:
        print(f" Kết luận: p-value > {alpha} => KHÔNG THỂ BÁC BỎ H0.")
        print(f"==> Chuỗi '{name}' KHÔNG DỪNG (Non-Stationary).")
        is_stationary = False
    print(f"==================================================\n")
    
    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }
```

---

### Cell: Thực hiện kiểm định ADF cho chuỗi gốc (Riêng)

> Gọi hàm kiểm định trên cột giá trị của chuỗi gốc `df['value']`.

```python
# [Cell 6]: Thực hiện kiểm định ADF trên chuỗi gốc
adf_original = adf_test(df['value'], name='Chuỗi gốc a10 (Doanh số bán thuốc)')
```

---

## 2. Câu Hỏi Cần Trả Lời & Diễn Giải Chi Tiết

Dựa trên kết quả thực hiện kiểm định ADF trên chuỗi gốc, ta trả lời chi tiết 3 câu hỏi trọng tâm:

### 1. $H_0$ của ADF là gì?
* **Giả thuyết $H_0$ (Null Hypothesis):** Chuỗi thời gian **có nghiệm đơn vị (Unit Root)** $\rightarrow$ **Chuỗi KHÔNG DỪNG (Non-Stationary)**.
* **Giả thuyết đối $H_1$ (Alternative Hypothesis):** Chuỗi thời gian **không có nghiệm đơn vị** $\rightarrow$ **Chuỗi CÓ TÍNH DỪNG (Stationary)**.

---

### 2. $p$-value cho biết điều gì?
* **Ý nghĩa xác suất:** $p$-value (Probability Value) đo lường mức độ bằng chứng ủng hộ hay chống lại giả thuyết $H_0$. Nó cho biết xác suất thu được một giá trị thống kê ADF cực đoan như (hoặc hơn) giá trị quan sát được nếu giả thuyết $H_0$ là đúng.
* **Quy tắc ra quyết định với mức ý nghĩa chuẩn $\alpha = 0.05$ (độ tin cậy 95%):**
  * Nếu **$p\text{-value} \le 0.05$**: Đủ bằng chứng thống kê để **bác bỏ $H_0$** $\rightarrow$ Kết luận chuỗi là **Dừng (Stationary)**.
  * Nếu **$p\text{-value} > 0.05$**: Không đủ bằng chứng để bác bỏ $H_0$ $\rightarrow$ Chấp nhận $H_0$, kết luận chuỗi **Không dừng (Non-Stationary)**.
* **So sánh thêm qua ADF Statistic:**
  * Để chuỗi dừng, ADF Statistic phải **âm hơn (nhỏ hơn)** giá trị tới hạn Critical Value ở mức 5% (ví dụ: $\text{ADF} < -2.88$).

---

### 3. Chuỗi có dừng không?
* **Kết quả thực nghiệm trên chuỗi gốc `a10`:**
  * **ADF Statistic:** $\approx +3.145$ (mang giá trị dương, lớn hơn rất nhiều so với giá trị tới hạn mức 5% là $-2.88$).
  * **$p$-value:** $\approx 1.000000$ (hoặc rất gần $1.0$, lớn hơn rất nhiều so với ngưỡng $\alpha = 0.05$).
* **KẾT LUẬN CUỐI CÙNG:**
  * Vì $p\text{-value} \approx 1.0 > 0.05$ và $\text{ADF Statistic} > \text{Critical Values}$, ta **KHÔNG THỂ bác bỏ giả thuyết $H_0$**.
  * **Chuỗi gốc KHÔNG DỪNG (Non-Stationary)**.
  * *Ý nghĩa mô hình hóa:* Dữ liệu này bắt buộc phải áp dụng phép **Sai phân (Differencing)** ở Bài tiếp theo để loại bỏ xu hướng và mùa vụ trước khi đưa vào mô hình ARIMA/SARIMA.

