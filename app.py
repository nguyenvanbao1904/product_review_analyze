from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        shopee_url = request.form.get("shopee_url")
        # Tạm thời in ra để debug
        print("Link người dùng nhập:", shopee_url)
        # Sau này gọi hàm crawl ở đây luôn
        return redirect(url_for("result", url=shopee_url))

    return render_template("index.html")

@app.route("/result")
def result():
    url = request.args.get("url")
    return f"Bạn đã nhập link: {url}"

if __name__ == "__main__":
    app.run(debug=True)
