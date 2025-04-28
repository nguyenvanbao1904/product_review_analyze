import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from services.preprocessing.utils import clean_text, load_abbreviations, load_stopwords

df = pd.read_csv("../../data/tiki_reviews_dataset.csv")

# Bo luon cot title vi no dua vao so sao nen khong co y nghia nhieu
df.drop(columns=['title'], inplace=True)

print(f"Tổng số dòng ban đầu: {len(df)}")

# ---Xoa cac dong bi trung
num_duplicates = df.duplicated().sum()
print(f"Số lượng dòng bị trùng lặp được tìm thấy: {num_duplicates}")
df = df.drop_duplicates()

print("Du lieu truoc khi xu ly missing_data")
print(len(df))
print(df.isnull().sum())

# ---Xu ly du lieu bi thieu

# Xoa cac dong khong co content vi no chem qua it khong dang ke (1 dong)
df = df.dropna(subset=["content"])

# Xoa cac dong khong co joined_time vi no chem qua it khong dang ke (27 dong)
df = df.dropna(subset=["joined_time"])

#Xu ly review_created_date, delivery
df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
df = df.dropna(subset=['delivery_date'])

df['review_created_date'] = pd.to_datetime(df['review_created_date'], errors='coerce')

df['review_gap'] = (df['review_created_date'] - df['delivery_date']).dt.days

df = df.drop(columns=["delivery_date", "review_created_date"])

print("Du lieu sau khi xu ly missing_data")
print(len(df))
print(df.isnull().sum())
# ---Chuan hoa du lieu
abbreviations_dict = load_abbreviations()
stopwords = load_stopwords()

df['thank_count'] = np.log1p(df['thank_count'])

df['purchased'] = df['purchased'].astype(int)

df['joined_time'] = df['joined_time'].str.extract(r'(\d+)').astype(float)

df = df[df['joined_time']<= 15]

df['total_review'] = np.log1p(df['total_review'])

df['review_gap'] = np.log1p(df['review_gap'])

df['is_photo'] = df['is_photo'].astype(int)

df['content'] = df['content'].apply(lambda x : clean_text(x, abbreviations_dict, stopwords))

print("Du lieu sau khi chuan hoa")
print(len(df))
print(df.isnull().sum())

df = df[df['content'] != ""]
print(len(df))
df = df[df['content'].str.len() > 1]
df = df.drop_duplicates()
print(len(df))
print(df.info())
# df.to_csv("../data/tiki_reviews_dataset_cleaned.csv", index=False, encoding='utf-8-sig')

