import sys
import os
sys.path.append(os.path.abspath('vits_japanese'))
from vits_japanese.model_deployment import text_to_speech

model_path = "model/best_model.pth"
config_path = "model/config.json"

# # Sử dụng hàm trên
text = "エラー。よく聞き取れませんでした|エラー。よく聞き取れませんでした"
out_path = "out.wav"

# Gọi hàm
text_to_speech(text, model_path, config_path, out_path)

