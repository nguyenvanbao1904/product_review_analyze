import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib

# Đọc dữ liệu
df = pd.read_csv("../../data/tiki_reviews_labeled.csv")

# Làm sạch các cột số (bỏ dấu chấm ngăn cách nghìn)
df['total_review'] = df['total_review'].str.replace('.', '', regex=False)
df['review_gap'] = df['review_gap'].str.replace('.', '', regex=False)
df['thank_count'] = df['thank_count'].str.replace('.', '', regex=False)

# Chuyển kiểu dữ liệu
df['thank_count'] = pd.to_numeric(df['thank_count'], errors='coerce')
df['total_review'] = pd.to_numeric(df['total_review'], errors='coerce')
df['review_gap'] = pd.to_numeric(df['review_gap'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Chọn đặc trưng và nhãn
X = df[["content", "thank_count", "purchased", "total_review", "is_photo", "review_gap", "rating"]]
y = df["is_pos"]

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hàm scale rating (nhân với alpha < 1 sau khi chuẩn hóa)
def scale_rating(x):
    alpha = 0.1
    return x * alpha

rating_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("weight", FunctionTransformer(scale_rating))
])

# Tiền xử lý các cột
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(), "content"),
        ("num", StandardScaler(), ["thank_count", "total_review", "review_gap"]),
        ("binary", "passthrough", ["purchased", "is_photo"]),
        ("rating", rating_pipe, ["rating"]),
    ])

# Pipeline tổng thể gồm preprocessing và mô hình Logistic Regression
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight='balanced', max_iter=500))
])

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred_custom = pipeline.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_custom)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_custom,
    labels=[-1, 0, 1],
    target_names=["Trung lập", "Tiêu cực", "Tích cực"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom, labels=[-1, 0, 1]))

# Phân tích tầm quan trọng đặc trưng
print("\n==> Tầm quan trọng của các đặc trưng (feature importance):")
result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
importances = result.importances_mean

# Lấy tên đặc trưng
try:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
except:
    feature_names = [f"feature_{i}" for i in range(len(importances))]

# In ra các đặc trưng quan trọng
for name, imp in zip(feature_names, importances):
    print(f"{name:30} : {imp:.5f}")

# Lưu mô hình nếu cần
# joblib.dump(pipeline, "../../model/my_model.pkl")