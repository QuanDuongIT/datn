import sys
import os
sys.path.append(os.path.abspath('vits_japanese'))
from vits_japanese.model_deployment import text_to_speech

# # Sử dụng hàm trên
text = "エラー。よく聞き取れませんでした|エラー。よく聞き取れませんでした"
out_path = "out.wav"

# Gọi hàm
text_to_speech(text,out_path)
