import sys
import os
import time
from flask import Flask, request, jsonify, send_file, render_template


sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_onnx,infer_long_text,split_text_by_sentences


with open('example1.txt', "r", encoding="utf-8") as file:
        content = file.read()


chunks = split_text_by_sentences(content,max_chars=200)

print(f"Số đoạn văn đã được cắt ra: {len(chunks)}")
print("Các đoạn đã cắt:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}:")

# infer_long_text(
#     content, 
#     model_path="vits2_pytorch/models/G_32000.onnx",
#     config_path="vits2_pytorch/models/config.json",
#     output_path = "out.wav"
# )