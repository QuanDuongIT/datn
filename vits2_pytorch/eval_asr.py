from infer_onnx import infer_long_text
import sys
sys.path.append('../eval_asr') 
from eval_audio import evaluate_tts_with_asr
import IPython.display as ipd

def synthesize_and_evaluate(
    text_path,
    model_path,
    config_path,
    audio_output_path="/content/audio.wav",
    whisper_model_size="medium",
    sampling_rate=16000
):
    """
    Tổng hợp giọng nói từ văn bản và đánh giá bằng Whisper ASR.

    Args:
        text_path (str): file Văn bản đầu vào.
        model_path (str): Đường dẫn đến mô hình ONNX.
        config_path (str): Đường dẫn đến file config.
        audio_output_path (str): File WAV sẽ được sinh ra.
        whisper_model_size (str): Mô hình Whisper dùng để nhận dạng.
        sampling_rate (int): Tốc độ lấy mẫu (Hz) của audio đầu ra.

    Returns:
        dict: Kết quả đánh giá bao gồm asr_text, phoneme gốc, phoneme asr, wer.
    """
    with open(text_path, "r", encoding="utf-8") as file:
      text = file.read()

    # Bước 1: Tổng hợp giọng nói
    infer_long_text(
        content=text,
        model_path=model_path,
        config_path=config_path,
        output_path=audio_output_path
    )

    # Bước 2: Đánh giá bằng Whisper
    result = evaluate_tts_with_asr(audio_output_path, text, whisper_model_size)

    # In kết quả
    print(f'| {whisper_model_size.upper()} |{"-"*80}')
    print(f"Văn bản gốc                              : {text}")
    print(f"Văn bản ASR                              : {result['asr_text']}")
    print(f"Phoneme gốc                              : {result['ground_truth_phonemes']}")
    print(f"Phoneme từ ASR                           : {result['asr_text_phonemes']}")
    print(f"Word Error Rate (WER)                    : {result['wer']*100:.2f}%")

    # Phát audio trong notebook
    return {
        **result,
        "audio": ipd.Audio(audio_output_path, rate=sampling_rate)
    }
