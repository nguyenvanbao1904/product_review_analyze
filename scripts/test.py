import pandas as pd
import joblib

from services.preprocessing.utils import clean_text_input

def print_samples(df, label_value, label_name, sample_frac):
    label_df = df[df['prediction'] == label_value]
    print(f"\n🔸 Một số đánh giá {label_name}:")

    if label_df.empty:
        print("- Không có đánh giá nào.")
    else:
        n_rows = label_df.shape[0]
        if n_rows >= 10:
            sampled_df = label_df.sample(frac=sample_frac, random_state=42)
        else:
            sampled_df = label_df.head(n_rows)  # In hết nếu quá ít
        for _, row in sampled_df.iterrows():
            print(f"- {row['content']}")

def get_recommendation(label_counts):
    positive_ratio = label_counts.get(1, 0)
    negative_ratio = label_counts.get(0, 0)

    # Nếu tỷ lệ tích cực >= 75% và tỷ lệ tiêu cực <= 10% -> Khuyến nghị mua
    if positive_ratio >= 75 and negative_ratio <= 10:
        return "Khuyến nghị mua: Sản phẩm được yêu thích và đáng để mua."

    # Nếu tỷ lệ tích cực từ 60% đến 75% -> Cân nhắc (cần thận trọng)
    elif positive_ratio >= 60 and positive_ratio < 75:
        return "Cân nhắc: Cần phân tích kỹ các đánh giá tiêu cực và trung lập. Cần thận trọng khi mua."

    # Nếu tỷ lệ tích cực < 60% hoặc tỷ lệ tiêu cực > 25% -> Không khuyến nghị mua
    elif positive_ratio < 60 or negative_ratio > 25:
        return "Không khuyến nghị mua: Sản phẩm có thể có vấn đề nghiêm trọng."
    else:
        return "Chưa đủ thông tin để đưa ra quyết định."
# Tải mô hình đã lưu
model = joblib.load("../model/my_model.pkl")

df = pd.read_csv("../data/test_data.csv")

df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
df = df.dropna(subset=['delivery_date'])
df['review_created_date'] = pd.to_datetime(df['review_created_date'], errors='coerce')
df['review_gap'] = (df['review_created_date'] - df['delivery_date']).dt.days
df['content'] = df['content'].apply(lambda x : clean_text_input(x))
df = df[df['content'] != ""]
features = ["content", "thank_count", "purchased", "total_review", "is_photo", "review_gap", "rating"]
X = df[features]

# Dự đoán
df['prediction'] = model.predict(X)
# Thống kê tỷ lệ các nhãn
label_counts = df['prediction'].value_counts(normalize=True) * 100
# Thống kê tỷ lệ các nhãn
label_counts = df['prediction'].value_counts(normalize=True) * 100

print("📊 Tỷ lệ dự đoán:")
print(f"Tích cực (1): {label_counts.get(1, 0):.2f}%")
print(f"Trung lập (-1): {label_counts.get(-1, 0):.2f}%")
print(f"Tiêu cực (0): {label_counts.get(0, 0):.2f}%")

# In các đánh giá ví dụ
print_samples(df, 1, "tích cực", sample_frac=0.1)
print_samples(df, -1, "trung lập", sample_frac=0.2)
print_samples(df, 0, "tiêu cực", sample_frac=0.2)

# Đưa ra khuyến nghị dựa trên tỷ lệ
recommendation = get_recommendation(label_counts)
print("\nKhuyến nghị:", recommendation)