import sys
import os
import time
from flask import Flask, request, jsonify, send_file, render_template


sys.path.append(os.path.abspath('vits2_pytorch'))
from vits2_pytorch.infer_onnx import infer_onnx

infer_onnx(
    text="エラー。よく聞き取れませんでした|エラー。よく聞き取れませんでした",
    model_path="vits2_pytorch/model/G_15000.onnx",
    config_path="vits2_pytorch/model/config.json",
    output_path = "o000000000000000ut.wav"
)