import requests

from services.scraping.tiki_reviews_scraper_detail import extract_ids_from_url, get_seller_id, get_reviews
if __name__ == "__main__":
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Referer": "https://tiki.vn/",
        "Accept": "application/json",
        "Accept-Language": "vi,en;q=0.9"
    }
    url = input("Nhap url: ")
    res = requests.get(url, headers=headers)
    count = 0
    for data in res.json()['data']:
        product_id, spid = extract_ids_from_url(data["url_path"])
        if product_id is not None and spid is not None:
            print(f"Extracted product_id = {product_id}, spid = {spid}")
            seller_id = get_seller_id(product_id, spid)
            print(f"seller_id = {seller_id}")
            file_path = "../../data/tiki_reviews_dataset.csv"

            get_reviews(product_id, spid, seller_id, file_path)
            count += 1
    print(f"Da lay danh gia cua {count} san pham")