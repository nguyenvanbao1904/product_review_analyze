import pandas as pd
import joblib
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("../data/tiki_reviews_sample_sorted.csv")
df1 = pd.read_csv("../data/tiki_reviews_labeled.csv")

df_sample = df.iloc[2904:3503].copy()


# Nối dữ liệu
df_concat = pd.concat([df1, df_sample], ignore_index=True)

df_concat.to_csv("../data/tiki_reviews_labeled.csv")

# /home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/venv/bin/python3.11 /home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/services/train_model/Logistic_Regression_Train.py
# Accuracy: 0.8513513513513513
#
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
#
# Process finished with exit code 0

# /home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/venv/bin/python3.11 /home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/services/train_model/Logistic_Regression_Train.py
# Accuracy: 0.8690773067331671
#
# Classification Report:
#               precision    recall  f1-score   support
#
#    Trung lập       S      0.58      0.47       151
#     Tiêu cực       0.94      0.90      0.92       655
#     Tích cực       0.95      0.90      0.92       798
#
#     accuracy                           0.87      1604
#    macro avg       0.76      0.79      0.77      1604
# weighted avg       0.89      0.87      0.88      1604
#
#
# Confusion Matrix:
# [[ 87  30  34]
#  [ 60 589   6]
#  [ 75   5 718]]
#
# Process finished with exit code 0

# /home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/venv/bin/python3.11 /home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/services/train_model/Logistic_Regression_Train.py
# Accuracy: 0.8654292343387471
#
# Classification Report:
#               precision    recall  f1-score   support
#
#    Trung lập       0.48      0.68      0.56       203
#     Tiêu cực       0.94      0.85      0.89       717
#     Tích cực       0.95      0.92      0.94       804
#
#     accuracy                           0.87      1724
#    macro avg       0.79      0.82      0.80      1724
# weighted avg       0.89      0.87      0.87      1724
#
#
# Confusion Matrix:
# [[139  30  34]
#  [103 610   4]
#  [ 49  12 743]]
#
# Process finished with exit code 0