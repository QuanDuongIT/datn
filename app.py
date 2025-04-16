import sys
import os
import time
from flask import Flask, request, jsonify, send_file, render_template

# Import model modules
sys.path.append(os.path.abspath('vits_japanese'))
from vits_japanese.model_deployment import text_to_speech, init_synthesizer

sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_onnx

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/vits2-japanese')
def vits2_page():
    return render_template('index.html', model_id='model1', title='🎙️ VITS2 Japanese')

@app.route('/vits2-experimental')
def vits2_experimental_page():
    return render_template('index.html', model_id='model3', title='🧬 VITS2 Experimental')

@app.route('/vits-old')
def vits_old_page():
    return render_template('index.html', model_id='model2', title='🧪 VITS Coqui/TTS')

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    data = request.get_json()
    text = data.get('text')
    model_id = data.get('model_id', 'model1')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    out_path = "out.wav"

    try:
        print(f'Running model: {model_id}')
        start_time = time.time()

        if model_id == 'model1':
            infer_onnx(
                text=text,
                model_path="vits2_pytorch/model/G_9000.onnx",
                config_path="vits2_pytorch/model/config.json",
                output_path=out_path
            )
        elif model_id == 'model3':
            infer_onnx(
                text=text,
                model_path="vits2_pytorch/model/G_ms_2000.onnx",
                config_path="vits2_pytorch/model/config_ms.json",
                output_path=out_path,
                sid=data.get('sid', None)
            )
        else:
            synthesizer = init_synthesizer("model/best_model.pth", "model/config.json", use_cuda=False)
            text_to_speech(text, out_path, synthesizer)

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
        webbrowser.open_new("http://127.0.0.1:5000/vits2-japanese")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.25, open_browser).start()

    app.run(debug=True)
