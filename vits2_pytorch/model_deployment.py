from infer_onnx import infer_onnx

infer_onnx(
    text="こんにちは世界、お元気ですか?",
    model_path="E:/datn/vits2_pytorch/model/G_ms_2000.onnx",
    config_path="E:/datn/vits2_pytorch/model/config_ms.json",
    output_path="output111.wav",
    sid=1
)
# from text import japanese

# # Văn bản tiếng Nhật
# text = "こんにちは、私はAIです。"

# # Chuyển văn bản tiếng Nhật thành IPA
# ipa_text = japanese.japanese_to_ipa(text)

# # In kết quả
# print(ipa_text)
