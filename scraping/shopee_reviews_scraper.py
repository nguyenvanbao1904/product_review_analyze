import requests
import pandas as pd
import re
import os
from datetime import datetime

def extract_ids_from_url(url):
    product_id_match = re.search(r"-p(\d+)", url)
    spid_match = re.search(r"spid=(\d+)", url)
    if product_id_match and spid_match:
        return int(product_id_match.group(1)), int(spid_match.group(1))
    else:
        raise ValueError("Không tìm được product_id hoặc spid từ URL.")

def unix_to_date(timestamp):
    try:
        return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ""

def get_seller_id(product_id, spid):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Referer": "https://tiki.vn/",
        "Accept": "application/json",
        "Accept-Language": "vi,en;q=0.9"
    }

    url = f"https://tiki.vn/api/v2/products/{product_id}?spid={spid}"

    res = requests.get(url, headers=headers)
    return res.json()["current_seller"]["id"]

def get_reviews(product_id, spid, seller_id, save_path):
    url = "https://tiki.vn/api/v2/reviews"
    params = {
        "limit": 5,
        "include": "comments,contribute_info,attribute_vote_summary",
        "sort": "score|desc,id|desc,stars|all",
        "page": 1,
        "spid": spid,
        "product_id": product_id,
        "seller_id": seller_id,
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://tiki.vn/",
        "Accept": "application/json",
        "Accept-Language": "vi,en;q=0.9"
    }

    reviews = []
    res = requests.get(url, headers=headers, params=params)

    if res.status_code == 200:
        datas = res.json()
        total_reviews = datas['reviews_count']
        total_pages = (total_reviews + params['limit'] - 1) // params['limit']
        print(f"Tổng số đánh giá: {total_reviews} | Tổng số trang: {total_pages}")

        for page in range(1, total_pages + 1):
            params['page'] = page
            res = requests.get(url, headers=headers, params=params)

            if res.status_code == 200:
                datas = res.json()
                print(f"Đang lấy trang {page}/{total_pages}")
                for data in datas['data']:
                    if data["content"].strip():
                        review = {
                            "product_id": product_id,
                            "spid": spid,
                            "title": data.get("title", ""),
                            "content": data.get("content", ""),
                            "rating": data.get("rating", ""),
                            "thank_count": data.get("thank_count", 0),
                            "purchased": data.get("created_by", {}).get("purchased", False),
                            "joined_time": data.get("created_by", {}).get("contribute_info", {}).get("summary", {}).get("joined_time", ""),
                            "total_review": data.get("created_by", {}).get("contribute_info", {}).get("summary", {}).get("total_review", 0),
                            "review_created_date": data.get("timeline", {}).get("review_created_date", ""),
                            "delivery_date": data.get("timeline", {}).get("delivery_date", ""),
                            "is_photo": data.get("is_photo", False)
                        }
                        reviews.append(review)
            else:
                print(f"Lỗi {res.status_code} khi lấy trang {page}")
                break

        # Sau vòng for: ghi một lần
        df = pd.DataFrame(reviews)
        if not os.path.exists(save_path):
            df.to_csv(save_path, index=False, mode='w', encoding='utf-8-sig', header=True)
        else:
            df.to_csv(save_path, index=False, mode='a', encoding='utf-8-sig', header=False)

        print(f"✅ Đã ghi {len(df)} đánh giá vào file {save_path}")
    else:
        print(f"Lỗi {res.status_code}: Không truy cập được API.")


if __name__ == "__main__":
    product_url = input("Nhập URL sản phẩm Tiki: ").strip()

    try:
        product_id, spid = extract_ids_from_url(product_url)
        seller_id = get_seller_id(product_id, spid)
        print(f"Extracted product_id = {product_id}, spid = {spid}")
        print(f"seller_id = {seller_id}")

        os.makedirs("../data", exist_ok=True)
        file_path = "../data/tiki_reviews_dataset.csv"

        get_reviews(product_id, spid, seller_id, file_path)
    except ValueError as e:
        print(f"Lỗi: {e}")
