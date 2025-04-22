import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("../../data/tiki_reviews_labeled.csv")

df['total_review'] = df['total_review'].str.replace('.', '', regex=False)
df['review_gap'] = df['review_gap'].str.replace('.', '', regex=False)
df['thank_count'] = df['thank_count'].str.replace('.', '', regex=False)


X = df[["content", "thank_count", "purchased", "total_review", "is_photo", "review_gap", "rating"]]
y = df["is_pos"]

df['thank_count'] = pd.to_numeric(df['thank_count'], errors='coerce')
df['total_review'] = pd.to_numeric(df['total_review'], errors='coerce')
df['review_gap'] = pd.to_numeric(df['review_gap'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')


# Chia dữ liệu với 80% train, 20% test và giữ tỷ lệ nhãn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(), "content"),
        ("num", StandardScaler(), ["thank_count", "total_review", "review_gap", "rating"]),  # StandardScaler cho cột số
        ("binary", "passthrough", ["purchased", "is_photo"])  # Giữ nguyên cột nhị phân
    ])

# Xây dựng pipeline với tiền xử lý và Logistic Regression
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight='balanced', max_iter=500))
])

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = pipeline.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Thêm chỉ số khác
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[-1, 0, 1],
    target_names=["Trung lập","Tiêu cực" , "Tích cực"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]))

# joblib.dump(pipeline, "../../model/my_model.pkl")