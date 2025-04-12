from infer_onnx import infer_onnx

infer_onnx(
    text="こんにちは世界、お元気ですか?",
    model_path="E:/datn/vits2_pytorch/model/G_6000.onnx",
    config_path="E:/datn/vits2_pytorch/model/config.json",
    output_path="output111.wav"
)