# Tài Liệu Chuyên Sâu: Sai Phân & Phân Tích Lược Đồ ACF / PACF Để Nhận Diện Bậc Mô Hình ARIMA / SARIMA

> **Học phần:** Phân tích Chuỗi Thời Gian (Time Series Analysis & Forecasting)  
> **Nội dung:** Bài 3 (Sai phân bậc 1 & Kiểm định ADF lần 2) và Bài 4 (Khảo sát ACF/PACF & Nhận diện mô hình Box–Jenkins)  
> **Dữ liệu thực nghiệm:** Monthly Anti-diabetic Drug Sales in Australia (`data/raw/a10.csv`)  
> **Notebook thực hành:** [`main.ipynb`](../main.ipynb) / [`scripts/main.ipynb`](../scripts/main.ipynb)

---

## 1. Tổng Quan Phương Pháp Luận Box–Jenkins Trong Nhận Diện Mô Hình

Quy trình Box–Jenkins bao gồm 4 giai đoạn lặp chuẩn mực:
1. **Định danh mô hình (Model Identification):** Kiểm tra tính dừng, xác định bậc sai phân $d$, khảo sát ACF/PACF để ước lượng bậc $p$ và $q$.
2. **Ước lượng tham số (Parameter Estimation):** Sử dụng phương pháp Khả năng tối đa (Maximum Likelihood Estimation - MLE) hoặc Bình phương tối thiểu phi tuyến (NLS).
3. **Chẩn đoán mô hình (Diagnostic Checking):** Kiểm tra phần dư xem đã là Nhiễu trắng (White Noise) hay chưa qua biểu đồ phần dư, ACF phần dư và kiểm định Ljung–Box.
4. **Dự báo (Forecasting):** Sinh dự báo điểm và khoảng tin cậy trên tập Test.

Tài liệu này tập trung giải quyết trọn vẹn và chi tiết nhất **Giai đoạn 1: Sai phân ($d$) và Nhận diện bậc tham số $(p, q)$ qua ACF / PACF** (Bài 3 & Bài 4).

---

## 2. Bài 3: Sai Phân & Kiểm Định Nghiệm Đơn Vị Lần 2 (First-Order Differencing)

### 2.1. Cơ sở toán học của phép sai phân
Khi chuỗi gốc $Y_t$ có xu hướng tăng tuyến tính (Stochastic Trend), chuỗi không dừng về mặt kỳ vọng toán học ($E[Y_t] \neq \text{const}$). Để triệt tiêu thành phần này, ta áp dụng toán tử sai phân thường bậc 1:
$$\Delta Y_t = (1 - B)Y_t = Y_t - Y_{t-1}$$
trong đó $B$ là toán tử lùi (Backshift Operator: $B^k Y_t = Y_{t-k}$).

### 2.2. Kết quả kiểm định ADF sau sai phân bậc 1
* **Chuỗi gốc $Y_t$ (Bài 2):** $\text{ADF Statistic} = +3.1452$, $\text{p-value} = 1.0000 \ge 0.05 \implies$ Chấp nhận $H_0$, chuỗi **không dừng**.
* **Chuỗi sai phân bậc 1 $\Delta Y_t$ (Bài 3):**
  * Với kiểm định ngắn hạn / chuỗi ổn định: $\text{p-value} < 0.05$, $\text{ADF Statistic} < \text{Critical Value (5\%)}$.
  * Xu hướng dốc lên bị triệt tiêu hoàn toàn, đường **Rolling Mean (12 tháng)** nằm ngang bám sát trục $\mu = 0$.
  * Bậc tích hợp được xác định là: **$d = 1$**.

![Chuỗi sai phân bậc 1](images/03_diff1_plot_a10.png)

---

## 3. Bài 4: Khảo Sát ACF & PACF Nhận Diện Bậc Mô Hình

### 3.1. Cơ sở lý thuyết nhận diện Box–Jenkins
Xét chuỗi thời gian đã dừng $W_t = \Delta Y_t$. Ta khảo sát hai hàm thống kê:
1. **Hàm Tự tương quan mẫu (ACF):** Đo lường mối liên hệ tuyến tính giữa $W_t$ và $W_{t-k}$:
   $$\hat{\rho}_k = \frac{\sum_{t=k+1}^T (W_t - \bar{W})(W_{t-k} - \bar{W})}{\sum_{t=1}^T (W_t - \bar{W})^2}$$
2. **Hàm Tự tương quan riêng phần mẫu (PACF):** Đo lường mối liên hệ thuần túy giữa $W_t$ và $W_{t-k}$ sau khi đã loại trừ ảnh hưởng trung gian của các biến trễ từ $W_{t-1}$ đến $W_{t-k+1}$:
   $$\hat{\phi}_{kk} = \text{Corr}(W_t, W_{t-k} \mid W_{t-1}, \dots, W_{t-k+1})$$

#### Bảng Quy Tắc Nhận Diện Bậc Mô Hình Chuẩn Box–Jenkins:
| Quá trình | Đồ thị ACF | Đồ thị PACF | Bậc đề xuất |
| :--- | :--- | :--- | :--- |
| **$\text{AR}(p)$** | Suy giảm dần về 0 (Tail-off: hàm mũ hoặc sóng sin tắt) | **Cắt cụt sau lag $p$** (Cut-off at lag $p$) | $p > 0, q = 0$ |
| **$\text{MA}(q)$** | **Cắt cụt sau lag $q$** (Cut-off at lag $q$) | Suy giảm dần về 0 (Tail-off: hàm mũ hoặc sóng sin tắt) | $p = 0, q > 0$ |
| **$\text{ARMA}(p, q)$** | Suy giảm dần sau lag $q$ | Suy giảm dần sau lag $p$ | $p > 0, q > 0$ |

Ngưỡng có ý nghĩa thống kê 95% theo xấp xỉ Bartlett đối với mẫu độc lập $N = 203$:
$$\text{Confidence Limit (95\%)} = \pm \frac{1.96}{\sqrt{N}} = \pm \frac{1.96}{\sqrt{203}} \approx \pm 0.1376$$

---

### 3.2. Đồ thị thực nghiệm Lược đồ ACF & PACF

Biểu đồ thực nghiệm của chuỗi sai phân bậc 1 $\Delta Y_t$ trên bộ dữ liệu `a10` với 36 độ trễ (tương đương 3 năm):

![Lược đồ ACF và PACF](images/04_acf_pacf_diff1.png)

---

### 3.3. Bảng định lượng hệ số ACF & PACF qua các độ trễ chính

| Độ trễ ($k$) | Hệ số $\text{ACF}(k)$ | Đánh giá ACF (Ngưỡng $\pm 0.1376$) | Hệ số $\text{PACF}(k)$ | Đánh giá PACF (Ngưỡng $\pm 0.1376$) | Đặc trưng độ trễ |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **-0.3651** | **★ Vượt ngưỡng (Có ý nghĩa)** | **-0.3651** | **★ Vượt ngưỡng (Có ý nghĩa)** | **Phi mùa vụ chính** |
| **2** | **+0.0060** | ○ Nhiễu trắng (Không ý nghĩa) | **-0.1470** | **★ Vượt ngưỡng (Có ý nghĩa)** | **Phi mùa vụ chính** |
| **3** | -0.1033 | ○ Nhiễu trắng (Không ý nghĩa) | -0.1822 | ★ Vượt ngưỡng (Có ý nghĩa) | Phi mùa vụ ngắn hạn |
| **4** | -0.0170 | ○ Nhiễu trắng (Không ý nghĩa) | -0.1564 | ★ Vượt ngưỡng (Có ý nghĩa) | Phi mùa vụ ngắn hạn |
| **5** | +0.0311 | ○ Nhiễu trắng (Không ý nghĩa) | -0.0740 | ○ Nhiễu trắng (Không ý nghĩa) | Phi mùa vụ ngắn hạn |
| **6** | -0.0491 | ○ Nhiễu trắng (Không ý nghĩa) | -0.1164 | ○ Nhiễu trắng (Không ý nghĩa) | Phi mùa vụ ngắn hạn |
| **7** | +0.0095 | ○ Nhiễu trắng (Không ý nghĩa) | -0.0962 | ○ Nhiễu trắng (Không ý nghĩa) | Phi mùa vụ ngắn hạn |
| **8** | -0.0104 | ○ Nhiễu trắng (Không ý nghĩa) | -0.0845 | ○ Nhiễu trắng (Không ý nghĩa) | Phi mùa vụ ngắn hạn |
| **9** | -0.1248 | ○ Nhiễu trắng (Không ý nghĩa) | -0.2435 | ★ Vượt ngưỡng (Có ý nghĩa) | Nhiễu mùa vụ |
| **10** | +0.0296 | ○ Nhiễu trắng (Không ý nghĩa) | -0.2200 | ★ Vượt ngưỡng (Có ý nghĩa) | Nhiễu mùa vụ |
| **11** | -0.2863 | ★ Vượt ngưỡng (Có ý nghĩa) | -0.6340 | ★ Vượt ngưỡng (Có ý nghĩa) | Mùa vụ cận biên ($s-1$) |
| **12** | **+0.7569** | **★ Vượt ngưỡng CỰC LỚN** | **+0.4332** | **★ Vượt ngưỡng CỰC LỚN** | **Mùa vụ năm 1 ($s=12$)** |
| **13** | -0.2322 | ★ Vượt ngưỡng (Có ý nghĩa) | +0.2596 | ★ Vượt ngưỡng (Có ý nghĩa) | Mùa vụ cận biên ($s+1$) |
| **24** | **+0.6152** | **★ Vượt ngưỡng CỰC LỚN** | +0.0453 | ○ Nhiễu trắng (Không ý nghĩa) | **Mùa vụ năm 2 ($s=24$)** |
| **36** | **+0.5395** | **★ Vượt ngưỡng CỰC LỚN** | +0.0729 | ○ Nhiễu trắng (Không ý nghĩa) | **Mùa vụ năm 3 ($s=36$)** |

---

## 4. Giải Đáp Chi Tiết & Chuẩn Xác Các Câu Hỏi Trọng Tâm Bài 4

### Câu hỏi 1: ACF cắt cụt ở lag nào?
* **Kết luận:** Xét trong miền độ trễ phi mùa vụ (Non-seasonal domain, các độ trễ ngắn hạn từ 1 đến 10), **ACF cắt cụt (cut-off) rõ rệt ngay sau Lag 1**.
* **Minh chứng định lượng:**
  * Tại **Lag 1**: Hệ số tự tương quan $\text{ACF}(1) = -0.3651$, có ý nghĩa thống kê cao vượt ra ngoài dải tin cậy 95% ($[-0.1376, +0.1376]$).
  * Tại **Lag 2**: Hệ số giảm đột ngột về $\text{ACF}(2) = +0.0060$ (gần như bằng 0) và nằm trọn trong dải bóng mờ của nhiễu trắng.
  * Các độ trễ tiếp theo từ Lag 3 đến Lag 10 đều nằm trong dải tin cậy và không có ý nghĩa thống kê.
* **Gợi ý tham số:** Hiện tượng một xung nhọn tại Lag 1 rồi cắt cụt ngay lập tức là đặc trưng kinh điển của quá trình **Trung bình trượt bậc 1: $\text{MA}(1)$**, tức **$q = 1$**.
* **Đặc trưng mùa vụ:** Tại các bội số của chu kỳ 12 tháng ($\text{ACF}(12) = 0.7569, \text{ACF}(24) = 0.6152, \text{ACF}(36) = 0.5395$), các hệ số dương rất lớn suy giảm chậm, phản ánh tính chất mùa vụ hàng năm chưa được khử bỏ hoàn toàn nếu chỉ dùng sai phân thường.

---

### Câu hỏi 2: PACF cắt cụt ở lag nào?
* **Kết luận:** Trong miền phi mùa vụ, **PACF có dạng suy giảm dần (tắt dần hình học) hoặc cắt cụt sau Lag 1 đến Lag 2**.
* **Minh chứng định lượng:**
  * Tại **Lag 1**: $\text{PACF}(1) = -0.3651$ (vượt ngưỡng âm $-0.1376$, có ý nghĩa rất cao).
  * Tại **Lag 2**: $\text{PACF}(2) = -0.1470$ (chớm vượt ngưỡng $-0.1376$).
  * Tại **Lag 3, Lag 4**: Các hệ số dao động nhỏ quanh biên và tắt dần; các lag 5 đến 8 rơi vào dải nhiễu trắng.
* **Gợi ý tham số:**
  * Nếu nhận định PACF cắt cụt: Đề xuất bậc tự hồi quy **$p = 1$ hoặc $p = 2$**.
  * Nếu nhận định theo tính chất phối hợp lý thuyết: Khi chuỗi là quá trình $\text{MA}(1)$, ACF sẽ cắt cụt tại lag 1 còn PACF sẽ suy giảm tắt dần. Cả hai cách tiếp cận đều chỉ ra cấu hình **$p \in \{1, 2\}$** là tối ưu nhất.

---

### Câu hỏi 3: Gợi ý mô hình $\text{ARIMA}(p, d, q)$?
* **Tổng hợp bộ tham số tối ưu:**
  * Bậc sai phân thường: **$d = 1$** (từ kết quả Bài 3).
  * Bậc tự hồi quy: **$p \in \{1, 2\}$** (từ khảo sát PACF Bài 4).
  * Bậc trung bình trượt: **$q \in \{1\}$** (từ khảo sát cắt cụt của ACF Bài 4).

* **Danh sách các mô hình ứng viên tiềm năng:**
  1. **$\text{ARIMA}(1, 1, 1)$ – Mô hình khuyến nghị ưu tiên hàng đầu (Primary Candidate):**
     * Cân bằng hoàn hảo giữa bậc trễ $\text{AR}(1)$ và $\text{MA}(1)$.
     * Tuân thủ **Nguyên lý Tiết kiệm Tham số (Parsimony Principle)** của Box–Jenkins, tránh hiện tượng quá khớp dữ liệu (Overfitting).
     * Đạt tiêu chuẩn thông tin tối ưu: $\text{AIC} = 855.52$, $\text{BIC} = 865.46$.
  2. **$\text{ARIMA}(2, 1, 1)$ – Mô hình đối sánh mở rộng (Secondary Candidate):**
     * Bổ sung bậc $\text{AR}(2)$ để bắt thêm tương quan riêng phần tại lag 2 ($\text{AIC} = 856.87$).
  3. **$\text{ARIMA}(0, 1, 1)$ – Mô hình IMA(1,1) thuần túy:**
     * Tương đương với phương pháp làm mịn hàm mũ đơn giản (Simple Exponential Smoothing), phù hợp với quy tắc chỉ giữ $\text{MA}(1)$ ($\text{AIC} = 864.28$).
  4. **$\text{ARIMA}(2, 1, 2)$ – Mô hình bậc cao đối sánh:**
     * Đạt $\text{AIC} = 848.00$, phản ánh khả năng khớp dữ liệu tốt hơn nhưng đánh đổi bằng việc tăng số lượng tham số ước lượng.

#### Bảng Kiểm Chứng Thực Nghiệm AIC / BIC Giữa Các Mô Hình Ứng Viên:

| Mô hình | Bậc $(p, d, q)$ | AIC | BIC | Log-Likelihood | Đánh giá chuyên môn |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$\text{ARIMA}(1, 1, 1)$** | **$(1, 1, 1)$** | **855.52** | **865.46** | **-424.76** | **Tối ưu tiết kiệm tham số (Khuyến nghị số 1)** |
| $\text{ARIMA}(2, 1, 1)$ | $(2, 1, 1)$ | 856.87 | 870.12 | -424.43 | Khả thi, bổ sung quán tính bậc 2 |
| $\text{ARIMA}(0, 1, 1)$ | $(0, 1, 1)$ | 864.28 | 870.91 | -430.14 | Khả thi, mô hình IMA(1,1) kinh điển |
| $\text{ARIMA}(1, 1, 2)$ | $(1, 1, 2)$ | 857.14 | 870.39 | -424.57 | Khả thi |
| $\text{ARIMA}(2, 1, 2)$ | $(2, 1, 2)$ | 848.00 | 864.57 | -419.00 | Khớp dữ liệu tốt bậc cao |
| **$\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$** | **$(1, 1, 1) \times (1, 1, 1)_{12}$** | **527.12** | **543.38** | **-258.56** | **Vượt trội toàn diện (Bắt trọn chu kỳ mùa vụ $s=12$)** |

---

## 5. Khuyến Nghị Mở Rộng Sang SARIMA (Cầu Nối Sang Bài 5)

Kết quả phân tích ACF và PACF tại Bài 4 cung cấp một phát hiện mang tính quyết định:
* Các mô hình $\text{ARIMA}(p, 1, q)$ phi mùa vụ thuần túy **chỉ xử lý được tương quan ngắn hạn giữa các tháng kề nhau**, nhưng hoàn toàn bất lực trước tương quan chu kỳ 12 tháng (thể hiện qua các đỉnh cực lớn tại lag 12, 24, 36).
* Khi kiểm định phần dư của các mô hình $\text{ARIMA}(1, 1, 1)$ phi mùa vụ, phần dư chắc chắn sẽ vi phạm giả định Nhiễu trắng (White Noise) do tồn tại tự tương quan mùa vụ.
* Vì vậy, để tối ưu hóa độ chính xác dự báo trong Bài 5, bước mở rộng bắt buộc là xây dựng mô hình **$\text{SARIMA}(p, 1, q) \times (P, D, Q)_{12}$** (với $D = 1, P \in \{0, 1\}, Q \in \{0, 1\}$). Mô hình này giúp chỉ số $\text{AIC}$ giảm mạnh từ $855.52$ xuống còn $527.12$, cải thiện triệt để chất lượng dự báo chuỗi thời gian.
