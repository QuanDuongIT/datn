import sys
import os
from flask import Flask, request, jsonify, send_file, render_template

sys.path.append(os.path.abspath('vits_japanese'))
from vits_japanese.model_deployment import text_to_speech, init_synthesizer

sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_onnx

from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)

# ✅ Khởi tạo Synthesizer một lần
synthesizer = init_synthesizer("model/best_model.pth", "model/config.json", use_cuda=False) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    data = request.get_json()
    text = data.get('text')
    model_id = data.get('model_id', 'model1')  # default to model1

    if not text:
        return jsonify({"error": "No text provided"}), 400

    out_path = "out.wav"

    try:
        print(f'training model: {model_id}')
        if model_id == 'model1':
            infer_onnx(
                text=text,
                model_path="vits2_pytorch/model/G_6000.onnx",
                config_path="vits2_pytorch/model/config.json",
                output_path=out_path
            )
        else:
            # ✅ Dùng lại synthesizer đã khởi tạo
            text_to_speech(text, out_path, synthesizer)
            
        if not os.path.exists(out_path):
            return jsonify({"error": "Audio file was not created."}), 500

        print(f"Audio file created at: {out_path}")
        return send_file(out_path, mimetype='audio/wav', as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

