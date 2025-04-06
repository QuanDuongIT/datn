from TTS.utils.synthesizer import Synthesizer

def text_to_speech(text, model_path, config_path, out_path):
    # Cung cấp các tham số cần thiết cho Synthesizer
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,          # Đường dẫn đến mô hình TTS
        tts_config_path=config_path,        # Đường dẫn đến cấu hình TTS
        use_cuda=False                       # Sử dụng CPU thay vì GPU (set True nếu muốn dùng GPU)
    )
    
    # Synthesize giọng nói và lưu vào file
    wav = synthesizer.tts(text)  # Phát sinh sóng âm thanh từ văn bản
    synthesizer.save_wav(wav, out_path)  # Lưu sóng âm thanh vào file
    print(f"Audio saved to {out_path}")



