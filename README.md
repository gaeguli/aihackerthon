# smart_carck_detection
A YOLO-based application that predicts the potential risk of building collapse from images.


# 📦 How to Use
⚠️ Make sure the OPENAI_API_KEY is registered as an environment variable. This project uses GPT-4.1 mini.

⚠️ It is recommended to install packages using uv for faster and more reliable setup.

⚠️ Currently, only Korean is supported.
```
git clone https://github.com/gaeguli/smart_crack_detection.git
cd smart_crack_detection
uv sync --frozen
uv run app.py
```

# How does it work?
1. Click the `Open Img` button to select a photo.
2. The trained YOLO model runs inference on the selected image.
3. Click the `Send Result` button to send YOLO’s prediction to ChatGPT for analysis, which will then suggest appropriate actions.
# Preview
<img src="img/img1.png">

<br/>

<img src="img/img2.png">

<br/>

<br/>

