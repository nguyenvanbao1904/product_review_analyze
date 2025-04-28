from datetime import datetime

from flask import Flask, render_template, request

from services.preprocessing.utils import clean_text_input
from services.scraping.tiki_reviews_scraper_detail import extract_ids_from_url, get_seller_id, get_reviews
import joblib
import pandas as pd
app = Flask(__name__)
def get_label_percent(df):
    return df['prediction'].value_counts(normalize=True) * 100

def get_recommendation(label_counts):
    positive_ratio = label_counts.get(1, 0)
    negative_ratio = label_counts.get(0, 0)

    if positive_ratio >= 80 and negative_ratio <= 10:
        return "Khuyến nghị mua: Sản phẩm được yêu thích và đáng để mua."
    elif positive_ratio < 65 or negative_ratio > 30:
        return "Không khuyến nghị mua: Sản phẩm có thể có vấn đề nghiêm trọng."
    elif positive_ratio >= 65 and negative_ratio <= 30:
        return "Cân nhắc: Cần phân tích kỹ các đánh giá tiêu cực và trung lập."
    else:
        return "Chưa đủ thông tin để đưa ra quyết định."

def scale_rating(x):
    alpha = 0.05
    return x * alpha

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        link = request.form["product_link"]
        try:
            product_id, spid = extract_ids_from_url(link)
            seller_id = get_seller_id(product_id, spid)
            print(f"Extracted product_id = {product_id}, spid = {spid}")
            print(f"seller_id = {seller_id}")

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"./tmp_csv/reviews_{product_id}_{now}.csv"
            get_reviews(product_id, spid, seller_id, save_path)

            df = pd.read_csv(save_path)
            df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
            df = df.dropna(subset=['delivery_date'])
            df['review_created_date'] = pd.to_datetime(df['review_created_date'], errors='coerce')
            df['review_gap'] = (df['review_created_date'] - df['delivery_date']).dt.days
            df['content'] = df['content'].apply(clean_text_input)
            df = df[df['content'] != ""]

            model = joblib.load("./model/my_model.pkl")

            features = ["content", "thank_count", "purchased", "total_review", "is_photo", "review_gap", "rating"]
            X = df[features]
            df['prediction'] = model.predict(X)

            label_counts = get_label_percent(df)
            recommendation = get_recommendation(label_counts)

            # Đưa 1 vài ví dụ
            examples = {
                "positive": df[df['prediction'] == 1]["content"].sample(
                    min(4, df[df['prediction'] == 1].shape[0])).tolist(),
                "neutral": df[df['prediction'] == -1]["content"].sample(
                    min(4, df[df['prediction'] == -1].shape[0])).tolist(),
                "negative": df[df['prediction'] == 0]["content"].sample(
                    min(4, df[df['prediction'] == 0].shape[0])).tolist()
            }

            return render_template(
                "result.html",
                label_counts=label_counts.to_dict(),
                recommendation=recommendation,
                examples=examples
            )

        except ValueError as e:
            print(f"Lỗi: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
