import argparse
import numpy as np
import onnxruntime
import torch
from scipy.io.wavfile import write
import time
import commons
import utils
from text import text_to_sequence
import textwrap
import os
import re
from tqdm import tqdm
from pydub import AudioSegment

def split_text_by_sentences(text, max_chars=200):
    # Tách câu bằng dấu chấm, dấu hỏi, hoặc dấu cảm thán
    sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    
    # Gom các câu lại thành đoạn không vượt quá max_chars
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def infer_onnx(text, model_path, config_path, output_path, sid=None, providers=None):
    """
    text: str - input text
    model_path: str - path to .onnx model
    config_path: str - path to config.json
    output_path: str - where to save .wav output
    sid: int | None - optional speaker ID
    providers: list | None - e.g., ["CUDAExecutionProvider"]
    """
    start_time = time.time()  # Bắt đầu đo thời gian

    
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers or ["CPUExecutionProvider"]
    )

    hps = utils.get_hparams_from_file(config_path)
    phoneme_ids = get_text(text, hps)
    text_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text_array.shape[1]], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    sid_array = np.array([int(sid)], dtype=np.int64) if sid is not None else None

    audio = model.run(
        None,
        {
            "input": text_array,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": sid_array,
        },
    )[0].squeeze((0, 1))

    write(filename=output_path, rate=hps.data.sampling_rate, data=audio)
    
    end_time = time.time()  # Kết thúc đo thời gian
    duration = end_time - start_time  # Tổng thời gian
    # print(f"⏱️ Time taken for synthesis: {duration:.2f} seconds")
    
    return output_path

def infer_long_text(content, model_path, config_path, output_path, sid=None, providers=None):
    # Chia nội dung thành các đoạn ngắn hơn (ví dụ: mỗi đoạn 200 ký tự)
    start_time = time.time()  # Bắt đầu đo thời gian
    
    # Chia nội dung thành các đoạn theo câu, mỗi đoạn tối đa 200 ký tự
    chunks = split_text_by_sentences(content, max_chars=200)

    # Tạo một đối tượng AudioSegment trống để ghép các đoạn âm thanh
    final_audio = AudioSegment.empty()

    # Duyệt qua từng đoạn văn bản và hiển thị thanh tiến trình
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Synthesis Progress", unit="chunk"):
        # Đặt tên tạm thời cho các tệp âm thanh
        temp_path = f"audio_part_{i}.wav"
        
        # Gọi hàm infer_onnx để tạo âm thanh từ đoạn văn bản
        infer_onnx(
            text=chunk,
            model_path=model_path,
            config_path=config_path,
            output_path=temp_path,
            sid=sid,
            providers=providers
        )
        
        # Thêm âm thanh từ đoạn vừa tạo vào final_audio
        final_audio += AudioSegment.from_wav(temp_path)
        
        # Xóa tệp âm thanh tạm thời sau khi sử dụng
        os.remove(temp_path)

    # Xuất kết quả cuối cùng ra tệp WAV
    final_audio.export(output_path, format="wav")
    end_time = time.time()  # Kết thúc đo thời gian
    duration = end_time - start_time  # Tổng thời gian
    print(f"⏱️ Time taken for synthesis: {duration:.2f} seconds")
    print(f"Final audio saved at: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.onnx)")
    parser.add_argument("--config-path", required=True, help="Path to model config (.json)")
    parser.add_argument("--output-wav-path", required=True, help="Path to write WAV file")
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize")
    parser.add_argument("--sid", required=False, type=int, help="Speaker ID to synthesize")
    args = parser.parse_args()

    infer_onnx(
        text=args.text,
        model_path=args.model,
        config_path=args.config_path,
        output_path=args.output_wav_path,
        sid=args.sid
    )


if __name__ == "__main__":
    main()

