import sys
import os
import time
from flask import Flask, request, jsonify, send_file, render_template


sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_onnx,infer_long_text

with open('example1.txt', "r", encoding="utf-8") as file:
        content = file.read()

infer_long_text(
    content, 
    model_path="vits2_pytorch/models/G_32000.onnx",
    config_path="vits2_pytorch/models/config.json",
    output_path = "out.wav"
)