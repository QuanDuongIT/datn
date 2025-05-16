import argparse
import numpy as np
import onnxruntime
import torch
from scipy.io.wavfile import write
import time
import commons
# import utils
from text import text_to_sequence
import textwrap
import os
import re
from tqdm import tqdm
import soundfile as sf 

import json
from types import SimpleNamespace

def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Chuyển dict lồng nhau thành object để dùng .data, .sampling_rate,...
    return json_to_namespace(data)

def json_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: json_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [json_to_namespace(i) for i in d]
    else:
        return d

def split_text_by_sentences(text, max_chars=100):
    """
    Tách văn bản tiếng Nhật theo câu và ghép các câu liền kề sao cho mỗi đoạn có < max_chars ký tự.
    
    Args:
        text (str): Văn bản đầu vào.
        max_chars (int): Số ký tự tối đa trong mỗi đoạn.
        
    Returns:
        List[str]: Danh sách các đoạn văn đã ghép.
    """
    # Tách câu theo dấu kết thúc
    raw_sentences = re.split(r'(?<=[。！？!?])', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_char_count = 0

    for sentence in sentences:
        char_count = len(sentence)
        if current_char_count + char_count <= max_chars:
            current_chunk.append(sentence)
            current_char_count += char_count
        else:
            if current_chunk:
                chunks.append("".join(current_chunk))
            current_chunk = [sentence]
            current_char_count = char_count

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks
    """
    Tách văn bản theo câu và ghép các câu liền kề sao cho mỗi đoạn có < max_words.
    
    Args:
        text (str): Văn bản đầu vào.
        max_words (int): Số từ tối đa trong mỗi đoạn (đếm theo khoảng trắng).
        
    Returns:
        List[str]: Danh sách các đoạn văn đã ghép.
    """
    # Bước 1: Tách câu theo dấu câu kết thúc
    raw_sentences = re.split(r'(?<=[。！？!?])', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    # Bước 2: Ghép các câu sao cho tổng số từ < max_words
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

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
    hps = get_hparams_from_file(config_path)
    # hps = utils.get_hparams_from_file(config_path)
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


def concatenate_wavs(wav_paths, output_path):
    """
    Ghép các file WAV thành một file duy nhất sử dụng soundfile.
    """
    audios = []
    sample_rate = None

    for path in wav_paths:
        data, sr = sf.read(path)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError("Sample rates do not match.")
        audios.append(data)

    final_audio = np.concatenate(audios, axis=0)
    sf.write(output_path, final_audio, sample_rate)

def infer_long_text(content, model_path, config_path, output_path, sid=None, providers=None):
    # Chia nội dung thành các đoạn ngắn hơn (ví dụ: mỗi đoạn 200 ký tự)
    start_time = time.time()  # Bắt đầu đo thời gian
    
    # Chia nội dung thành các đoạn theo câu, mỗi đoạn tối đa 200 ký tự
    chunks = split_text_by_sentences(content, max_chars=300)

    temp_files = []

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
        
        # Lưu lại các file tạm thời
        temp_files.append(temp_path)

    # Ghép tất cả các đoạn âm thanh lại với nhau
    concatenate_wavs(temp_files, output_path)

    # Xóa các file tạm
    for f in temp_files:
        os.remove(f)

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

