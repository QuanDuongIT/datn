import time
from TTS.utils.synthesizer import Synthesizer

# Khởi tạo sẵn Synthesizer (chỉ làm 1 lần)
def init_synthesizer(model_path, config_path, use_cuda=False):
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=use_cuda
    )
    return synthesizer


def text_to_speech(text, out_path, synthesizer):
    start_time = time.time()  # Bắt đầu đo thời gian

    # Chuyển văn bản thành sóng âm
    wav = synthesizer.tts(text)
    
    # Lưu sóng âm thanh vào file
    synthesizer.save_wav(wav, out_path)

    end_time = time.time()  # Kết thúc đo thời gian
    duration = end_time - start_time  # Tổng thời gian

    print(f"Audio saved to {out_path}")
    print(f"⏱️ Time taken for synthesis: {duration:.2f} seconds")
