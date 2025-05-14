import whisper
import pyopenjtalk
from jiwer import wer

def evaluate_tts_with_asr(audio_path, ground_truth_text, whisper_model_size="medium"):
    """
    Đánh giá chất lượng TTS bằng Whisper ASR + WER trên phoneme.

    Args:
        audio_path (str): Đường dẫn đến file âm thanh .wav.
        ground_truth_text (str): Văn bản gốc dùng để so sánh.
        whisper_model_size (str): Kích cỡ mô hình Whisper ("tiny", "base", "small", "medium", "large").

    Returns:
        dict: Gồm văn bản ASR, phonemes gốc, phonemes ASR và WER.
    """
    model = whisper.load_model(whisper_model_size)
    result = model.transcribe(audio_path)
    asr_text = result["text"].strip()

    # Chuyển sang phoneme
    ground_truth_phonemes = pyopenjtalk.g2p(ground_truth_text.strip(), kana=False)
    asr_text_phonemes = pyopenjtalk.g2p(asr_text, kana=False)

    gt_phoneme_str = " ".join(ground_truth_phonemes.strip().split())
    asr_phoneme_str = " ".join(asr_text_phonemes.strip().split())

    error_rate = wer(gt_phoneme_str, asr_phoneme_str)

    return {
        "asr_text": asr_text,
        "ground_truth_phonemes": ground_truth_phonemes,
        "asr_text_phonemes": asr_text_phonemes,
        "wer": error_rate
    }

