import sys
import os
from flask import Flask, request, jsonify, send_file, render_template, Response

sys.path.append(os.path.abspath('vits_japanese'))
from vits_japanese.model_deployment import text_to_speech

model_path = "src/model/best_model.pth"
config_path = "src/model/config.json"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Endpoint nhận văn bản và chuyển thành giọng nói
@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    # Nhận dữ liệu từ body request
    data = request.get_json()
    text = data.get('text')
    
    # Kiểm tra xem văn bản có hợp lệ không
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tạo đường dẫn tệp âm thanh đầu ra
    out_path = "out.wav"
    
    try:
        # Chuyển văn bản thành giọng nói
        text_to_speech(text, model_path, config_path, out_path)
        
        # Kiểm tra xem tệp âm thanh có tồn tại không
        if not os.path.exists(out_path):
            return jsonify({"error": "Audio file was not created."}), 500

        # In thông tin debug về tệp âm thanh
        print(f"Audio file created at: {out_path}")

        # Trả về tệp âm thanh dưới dạng file response
        return send_file(out_path, mimetype='audio/wav', as_attachment=False)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
