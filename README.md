# Phát hiện gian lận thẻ tín dụng

Bài toán **phân loại nhị phân** trên dataset giao dịch thẻ tín dụng: dự đoán một giao dịch là **bình thường (0)** hay **gian lận (1)**, xử lý mạnh mất cân bằng lớp bằng undersampling và so sánh hai cấu hình mô hình.

> **Điểm cốt lõi**: dữ liệu chỉ có ~0.17% giao dịch gian lận, nên accuracy không phải thước đo phù hợp — project tập trung vào **Precision / Recall / F1 / ROC-AUC** của lớp gian lận.

---

## Kết quả

| Version | Model | Sampling | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|-------|----------|----------|-----------|--------|----|---------|
| **v1** | Logistic Regression | 1:1 (355 / 355) | 0.9566 | 0.0337 | **0.9068** | 0.0651 | **0.9743** |
| **v2** ✅ | Random Forest | 10:1 (3550 / 355) | **0.9987** | **0.5793** | 0.8051 | **0.6738** | 0.9742 |

Confusion matrix trên tập test (70,932 giao dịch, 118 gian lận):

```
v1 — LogisticRegression          v2 — RandomForest
[[67750,  3064],                 [[70745,    69],
 [   11,   107]]                  [   23,    95]]
```

**Nhận xét**

- **v1** bắt được gần như toàn bộ gian lận (recall 91%) nhưng đánh dấu nhầm 3,064 giao dịch hợp lệ → precision chỉ 3.4%, không dùng được trong thực tế.
- **v2** giảm false positive gần 44 lần (3,064 → 69), precision tăng lên 58%, F1 tăng hơn 10 lần. Đây là cấu hình được chọn.
- ROC-AUC của hai mô hình gần như bằng nhau (~0.974) → thứ hạng xác suất tương đương, khác biệt nằm ở **ngưỡng quyết định** và **khả năng tách biên**.

---

## Kiến trúc pipeline

```
CSV giao dịch (284,807 × 31)
    │
    ▼
┌──────────────────┐   Missing values → 0
│  Data Cleaning   │   Duplicates 1081 → drop
└──────────────────┘   Time/Amount ≥ 0, Class ∈ {0,1}
    │
    ▼
┌──────────────────┐   Amount / Time / V1..V28
│       EDA        │   Univariate + correlation matrix
└──────────────────┘
    │
    ▼
┌──────────────────┐   train_test_split(stratify=y, 75/25)
│   Train / Test   │
└──────────────────┘
    │
    ▼
┌──────────────────┐   StandardScaler (fit trên train)
│ Feature Scaling  │
└──────────────────┘
    │
    ▼
┌──────────────────────────┐   v1: RandomUnderSampler(1:1)
│  Imbalance Handling      │   v2: RandomUnderSampler(0.1)
└──────────────────────────┘
    │
    ▼
┌──────────────────┐   LogisticRegression / RandomForest
│   Model Train    │
└──────────────────┘
    │
    ▼
   Evaluate: Confusion Matrix → Precision / Recall / F1 → ROC-AUC
```

---

## Dataset

| Thuộc tính | Giá trị |
|------------|---------|
| Số dòng gốc | 284,807 |
| Số cột | 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Missing values | 0 |
| Duplicate rows | 1,081 → xoá còn **283,726** dòng |
| Giao dịch bình thường | 283,253 (99.83%) |
| Giao dịch gian lận | 473 (0.17%) |
| `Time` | 0 – 172,792 giây (~48 giờ), mean 94,811 |
| `Amount` | 0 – 25,691.16, mean 88.47, median 22.00 |

- `V1`–`V28` là các thành phần **PCA đã ẩn danh hoá** từ feature gốc (không giải thích được ngữ nghĩa).
- `Time` = số giây từ giao dịch đầu tiên; `Amount` = giá trị giao dịch.
- Tập test sau split: **70,932 dòng**, trong đó **118 gian lận** (giữ nguyên tỷ lệ nhờ `stratify`).

---

## Các bước xử lý chính

### 1. Data Cleaning
- Kiểm tra missing values → không có.
- Kiểm tra & xoá duplicate (`drop_duplicates(inplace=True)`), 1,081 dòng bị loại.
- Kiểm tra kiểu dữ liệu → toàn bộ đã là numeric.
- Kiểm tra giá trị không hợp lệ: `Time.min() = 0`, `Amount.min() = 0` → không có giá trị âm.
- Validate target: `Class.unique() = [0, 1]` → chỉ 2 lớp.

### 2. EDA
- **Univariate**: `describe()` cho `Amount` và `Time`; histogram `Amount` cho thấy phân phối **lệch phải mạnh** (mean 88 >> median 22, max 25,691).
- **Multivariate**: correlation matrix giữa các feature — `V1`–`V28` gần như độc lập với nhau (đúng kỳ vọng của PCA), `Time` tương quan yếu với `V3` (-0.42) và `V5` (0.17).

### 3. Feature Scaling
- `StandardScaler` **fit trên train, transform trên test** để tránh data leakage.
- `Amount` và `Time` có scale rất khác `V*` nên scaling là bắt buộc với Logistic Regression.

### 4. Xử lý mất cân bằng
- **v1**: `RandomUnderSampler(random_state=42)` → 355 / 355 (cân bằng 1:1), mất nhiều dữ liệu lớp đa số.
- **v2**: `RandomUnderSampler(sampling_strategy=0.1)` → 3,550 / 355 (tỷ lệ 10:1), giữ nhiều thông tin hơn cho cây quyết định.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Ngôn ngữ | Python ≥ 3.10 |
| Xử lý dữ liệu | pandas |
| Modeling | scikit-learn (`LogisticRegression`, `RandomForestClassifier`) |
| Imbalance | imbalanced-learn (`RandomUnderSampler`) |
| Trực quan hoá | matplotlib, seaborn |
| Môi trường | Jupyter Notebook / PyCharm (`ipykernel`) |
| Quản lý package | uv (`pyproject.toml`, `uv.lock`) |

---

## Cài đặt và chạy

### 1. Cài dependencies

```bash
git clone https://github.com/<your-username>/credit_card_fraud.git
cd credit_card_fraud

uv sync            # khuyến nghị
# hoặc: pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt file CSV vào thư mục `data/`:

```
data/creditcard.csv
```

Notebook đọc dữ liệu bằng đường dẫn tương đối `data/creditcard.csv`, nên cần chạy từ thư mục gốc project.
Dataset gốc: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### 3. Chạy notebook

```bash
jupyter notebook Credit_card.ipynb
```

Hoặc mở trực tiếp `Credit_card.ipynb` trong PyCharm với kernel từ `.venv`.

Chạy tuần tự từ trên xuống: Cleaning → EDA → Split → Scaling → Sampling → Train → Evaluate.

> ⚠️ Cell **Version-2** dùng lại biến `X_train_scaled` từ phần trước, nên phải chạy hết các cell phía trên thì phần này mới chạy đúng.

---

## Cấu trúc project

```
credit_card_fraud/
├── data/
│   └── creditcard.csv        # dataset gốc (284,807 × 31)
├── Credit_card.ipynb         # notebook chính: cleaning → EDA → model → evaluate
├── pyproject.toml            # khai báo dependencies
├── uv.lock                   # lock file của uv
├── README_example.md         # template tham khảo cho README
└── README.md
```

---

## Hướng phát triển

- [ ] Bổ sung **SMOTE / ADASYN** để so sánh với undersampling (tránh mất dữ liệu lớp đa số)
- [ ] Tối ưu ngưỡng quyết định theo **precision-recall curve** thay vì ngưỡng 0.5 mặc định
- [ ] Thêm **XGBoost / LightGBM** và `class_weight='balanced'` làm baseline thứ ba
- [ ] Cross-validation (Stratified K-Fold) thay cho single split để kết quả ổn định hơn
- [ ] Vẽ precision-recall curve + feature importance của Random Forest
- [ ] Đóng gói pipeline (`sklearn.Pipeline`) và lưu model bằng `joblib`
- [ ] Tách code thành module `.py` + CLI để chạy lại toàn bộ thí nghiệm

---

## Lưu ý

- Dataset cực kỳ mất cân bằng (473 gian lận / 283,726 dòng) → **accuracy cao không có ý nghĩa**. Một model đoán "không gian lận" cho mọi giao dịch đã đạt 99.83%.
- Chỉ số đáng quan tâm là **Recall** (bỏ sót gian lận) và **Precision** (làm phiền khách hàng), hoặc F1 để cân bằng hai phía.
- `V1`–`V28` đã qua PCA nên không thể diễn giải trực tiếp; feature importance chỉ mang ý nghĩa thống kê.
