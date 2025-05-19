import sys
import os
import time
from flask import Flask, request, jsonify, send_file, render_template

# Thêm đường dẫn thư mục chứa mã nguồn VITS2
sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_long_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', model_id='model1', title='VITS Japanese')

@app.route('/vits2-japanese')
def vits2_page():
    return render_template('index.html', model_id='model1', title='VITS Japanese')

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    data = request.get_json()
    text = data.get('text')
    model_id = data.get('model_id', 'model1')
    voice = data.get('voice', 'female')  # Default: female

    if not text:
        return jsonify({"error": "No text provided"}), 400

    out_path = "out.wav"

    try:
        print(f'Running model: {model_id}, Voice: {voice}')
        start_time = time.time()

        if model_id == 'model1':
            # Chọn model tương ứng với giọng
            if voice == 'female':
                model_path = "vits2_pytorch/models/G_32000.onnx"
            elif voice == 'male':
                model_path = "vits2_pytorch/models/G_5000.onnx"
            else:
                return jsonify({"error": "Unsupported voice type."}), 400

            infer_long_text(
                content=text,
                model_path=model_path,
                config_path="vits2_pytorch/models/config.json",
                output_path=out_path
            )

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
