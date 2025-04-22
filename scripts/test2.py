import pandas as pd
import joblib
from services.preprocessing.utils import clean_text_input

# Đánh giá giả lập: 5 sao nhưng chê thậm tệ
sample_review = {
    "content": "Sản phẩm tệ kinh khủng. Vừa mở hộp đã hỏng, giao hàng trễ, chăm sóc khách hàng tệ hại, tôi rất thất vọng.",
    "thank_count": 0,
    "purchased": True,
    "total_review": 3,
    "is_photo": False,
    "review_gap": 2,
    "rating": 4
}

# Tạo DataFrame
df = pd.DataFrame([sample_review])
df['content'] = df['content'].apply(lambda x: clean_text_input(x))
model = joblib.load("../model/my_model.pkl")
# Dự đoán
X = df[["content", "thank_count", "purchased", "total_review", "is_photo", "review_gap", "rating"]]
df['prediction'] = model.predict(X)

# In kết quả
print("📝 Nội dung đánh giá:")
print(df['content'].iloc[0])
print("\n🔍 Dự đoán của mô hình:", "Tích cực" if df['prediction'].iloc[0] == 1 else "Trung lập" if df['prediction'].iloc[0] == -1 else "Tiêu cực")
