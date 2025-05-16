import sys
import os
import time
from flask import Flask, request, jsonify, send_file, render_template

sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_onnx, infer_long_text

with open('example1.txt', "r", encoding="utf-8") as file:
    content = file.read()

app = Flask(__name__)

# Đổi route '/' để trả về index.html thay vì home.html
@app.route('/')
def home():
    return render_template('index.html', model_id='model1', title='VITS Japanese')

# Giữ lại chỉ trang cho model1
@app.route('/vits2-japanese')
def vits2_page():
    return render_template('index.html', model_id='model1', title='VITS Japanese')

# Xoá route cho model3 và model2

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    data = request.get_json()
    text = data.get('text')
    model_id = data.get('model_id', 'model1')  # Mặc định model1

    if not text:
        return jsonify({"error": "No text provided"}), 400

    out_path = "out.wav"

    try:
        print(f'Running model: {model_id}')
        start_time = time.time()

        if model_id == 'model1':  # Chỉ xử lý model1
            infer_long_text(
                content=text,
                model_path="vits2_pytorch/models/G_32000.onnx",
                config_path="vits2_pytorch/models/config.json",
                output_path=out_path
            )

        # Loại bỏ phần xử lý cho model3 và model2

        exec_time = round(time.time() - start_time, 2)

        if not os.path.exists(out_path):
            return jsonify({"error": "Audio file was not created."}), 500

        response = send_file(out_path, mimetype='audio/wav', as_attachment=False)
        response.headers['X-Execution-Time'] = str(exec_time)
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.25, open_browser).start()

    app.run(debug=True)
