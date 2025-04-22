import emoji
import re
from pyvi import ViTokenizer
def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

# Nên load 1 lần bên ngoài hàm nếu tái sử dụng nhiều
def load_abbreviations():
    abbreviations = {}
    with open("/home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/data/abbreviations.txt", 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                key, value = line.strip().split('|')
                abbreviations[key.strip()] = value.strip()
    return abbreviations

def load_stopwords():
    with open('/home/nguyenvanbao/PhanTichDuLieu/product_review_analyze/data/stopwords.txt', "r", encoding="utf-8") as f:
        stopwords = {line.strip() for line in f if line.strip()}
    return stopwords

def clean_text(text, abbreviations_dict, stopwords):
    text = text.lower()
    text = remove_emoji(text)
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r'[^\w\sàáạảãăắằẵặẳâấầẫẩậèéẹẻẽêếềễểệìíịỉĩòóọỏõôốồỗổộơớờỡởợùúụủũưứừữửựỳýỵỷỹđ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = ViTokenizer.tokenize(text)

    words = text.split()
    # Sử dụng tham số abbreviations_dict đã truyền vào
    processed_words = [abbreviations_dict.get(word, word) for word in words]
    text = " ".join(processed_words)

    words = text.split()
    filtered = [word for word in words if word not in stopwords]
    return " ".join(filtered)

def clean_text_input(text):
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\w+)( \1){3,}\b', r'\1', text)
    if is_gibberish(text):
        return ""
    return text

def is_gibberish(text):
    """
    Xem xét chuỗi có phải là rác: quá ngắn, toàn từ vô nghĩa, hoặc lặp linh tinh
    """
    text = text.strip().lower()
    if len(text) < 2:
        return True

    # Loại bỏ dấu và số để dễ xử lý
    stripped = re.sub(r'[^\w\s]', '', text)
    words = stripped.split()

    # Nếu có ít hơn 3 từ thật sự → rác
    if len(words) < 2:
        return True

    # Nếu có quá nhiều từ giống nhau → rác
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:
        return True

    # Nếu toàn là chuỗi không có nguyên âm (a, e, i, o, u) → rác
    if not re.search(r'[aeiouáéíóúăâêôơư]', text):
        return True

    return False