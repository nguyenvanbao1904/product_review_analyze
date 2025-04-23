import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

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


def scale_rating(x):
    # Chọn hệ số giảm trọng số cho rating
    alpha = 0.3
    return x* alpha

def boost_text_weight(x):
    beta = 10
    return x * beta

text_pipeline = make_pipeline(
    TfidfVectorizer(),
    FunctionTransformer(boost_text_weight)
)

rating_pipe = Pipeline([
    ("scale", StandardScaler()),                            # scale về ~N(0,1)
    ("weight", FunctionTransformer(scale_rating))     # nhân với alpha < 1
])
preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_pipeline, "content"),
        ("num", StandardScaler(), ["thank_count", "total_review", "review_gap"]),  # StandardScaler cho cột số
        ("binary", "passthrough", ["purchased", "is_photo"]),  # Giữ nguyên cột nhị phân
        ("rating", rating_pipe, ["rating"]),
    ])

# Xây dựng pipeline với tiền xử lý và Logistic Regression
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight='balanced', max_iter=500))
])

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Dự đoán trên tập test
y_probs = pipeline.predict_proba(X_test)
neg_probs = y_probs[:, 1]
threshold_neg = 0.55
y_pred_custom = []

for i in range(len(y_probs)):
    # Nếu xác suất tiêu cực cao vượt ngưỡng → gán là 0
    if neg_probs[i] >= threshold_neg:
        y_pred_custom.append(0)
    else:
        # Gán nhãn có xác suất cao nhất trong 2 class còn lại
        probs = y_probs[i]
        max_index = np.argmax([probs[0], probs[2]])  # Chỉ so giữa -1 và 1
        y_pred_custom.append([-1, 1][max_index])


# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_custom)
print("Accuracy:", accuracy)

# Thêm chỉ số khác
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_custom,
    labels=[-1, 0, 1],
    target_names=["Trung lập","Tiêu cực" , "Tích cực"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom, labels=[-1, 0, 1]))


# 👉 Phân tích tầm quan trọng các đặc trưng
from sklearn.inspection import permutation_importance

print("\n==> Tầm quan trọng của các đặc trưng (feature importance):")
result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
importances = result.importances_mean

# Với ColumnTransformer, bạn cần lấy tên từng feature đúng theo thứ tự
try:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
except:
    feature_names = [f"feature_{i}" for i in range(len(importances))]  # fallback nếu không có tên

for name, imp in zip(feature_names, importances):
    print(f"{name:30} : {imp:.5f}")


#Accuracy: 0.8513513513513513

# Classification Report:
#               precision    recall  f1-score   support
#
#    Trung lập       0.40      0.57      0.47       119
#     Tiêu cực       0.85      0.82      0.84       201
#     Tích cực       0.96      0.90      0.93       790
#
#     accuracy                           0.85      1110
#    macro avg       0.73      0.76      0.74      1110
# weighted avg       0.88      0.85      0.86      1110
#
#
# Confusion Matrix:
# [[ 68  21  30]
#  [ 33 165   3]
#  [ 70   8 712]]

joblib.dump(pipeline, "../../model/my_model.pkl")