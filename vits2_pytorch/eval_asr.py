from infer_onnx import infer_long_text
import sys
sys.path.append('../eval_asr') 
from eval_audio import evaluate_tts_with_asr
import IPython.display as ipd
from get_loss import LogAnalyzer
import os
import re

def synthesize_and_evaluate(
    text_path,
    model_path,
    config_path,
    audio_output_path="/content/audio.wav",
    whisper_model_size="medium",
    sampling_rate=16000
):
    """
    Tổng hợp giọng nói từ văn bản và đánh giá bằng Whisper ASR,
    đồng thời in log từ checkpoint tương ứng nếu có.

    Args:
        text_path (str): Đường dẫn đến file văn bản đầu vào.
        model_path (str): Đường dẫn đến mô hình ONNX (G_<step>.onnx).
        config_path (str): Đường dẫn đến file config.
        audio_output_path (str): Đường dẫn file âm thanh đầu ra.
        whisper_model_size (str): Kích thước mô hình Whisper.
        sampling_rate (int): Tần số lấy mẫu của audio.

    Returns:
        dict: Kết quả đánh giá và audio phát lại.
    """

    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tổng hợp giọng nói
    infer_long_text(
        content=text,
        model_path=model_path,
        config_path=config_path,
        output_path=audio_output_path
    )

    # Đánh giá bằng Whisper
    result = evaluate_tts_with_asr(audio_output_path, text, whisper_model_size)
    text_clean = text.strip().replace('\n', ' ')
    
    # Hiển thị kết quả đánh giá
    print(f'\n| {whisper_model_size.upper()} |{"-"*80}')
    print(f"Văn bản gốc                              : {text_clean}")
    print(f"Văn bản ASR                              : {result['asr_text']}")
    print(f"Phoneme gốc                              : {result['ground_truth_phonemes']}")
    print(f"Phoneme từ ASR                           : {result['asr_text_phonemes']}")
    print(f"Word Error Rate (WER)                    : {result['wer']*100:.2f}%")

    # Tự động xác định thư mục log từ model_path
    log_dir = os.path.dirname(model_path)

    # Trích xuất step từ tên file (ví dụ: G_36000.onnx)
    match = re.search(r'G_(\d+)\.onnx', os.path.basename(model_path))
    if match:
        step = int(match.group(1))
        try:
            analyzer = LogAnalyzer(log_dir)
            analyzer.search_logs_by_checkpoint(step)
        except Exception as e:
            print(f"Lỗi khi phân tích log: {e}")
    else:
        print("Không thể xác định step từ tên file mô hình.")

    # Trả về kết quả và audio
    return {
        **result,
        "audio": ipd.Audio(audio_output_path, rate=sampling_rate)
    }