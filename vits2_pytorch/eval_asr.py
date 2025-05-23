import os
import re
import csv
from infer_onnx import infer_long_text
import sys
sys.path.append('../eval_asr') 
from eval_audio import evaluate_tts_with_asr
import IPython.display as ipd
from get_loss import LogAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

def synthesize_and_evaluate(
    text_path,
    model_path,
    config_path,
    audio_output_path="/content/audio.wav",
    whisper_model_size="medium",
    sampling_rate=16000,
    csv_path=None
):
    from get_loss import LogAnalyzer  # Đảm bảo đã import
    import csv

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

    print(f'\n| {whisper_model_size.upper()} |{"-"*80}')
    print(f"Văn bản gốc         : {text_clean}")
    print(f"Văn bản ASR         : {result['asr_text']}")
    print(f"Phoneme gốc         : {result['ground_truth_phonemes']}")
    print(f"Phoneme từ ASR      : {result['asr_text_phonemes']}")
    print(f"Word Error Rate     : {result['wer']*100:.2f}%")

    log_dir = os.path.dirname(os.path.dirname(model_path))
    step = None
    loss_data = {
        "loss_g_total": None,
        "loss_d_total": None,
        "loss_mel": None,
        "loss_fm": None,
        "loss_g_kl": None
    }

    # Trích xuất step
    match = re.search(r'G_(\d+)\.onnx', os.path.basename(model_path))
    if match:
        step = int(match.group(1))
        try:
            analyzer = LogAnalyzer(log_dir)
            loss_data = analyzer.search_logs_by_checkpoint(step)
        except Exception as e:
            print(f"Lỗi khi phân tích log: {e}")
    else:
        print("Không thể xác định step từ tên file mô hình.")

    # Tạo file CSV nếu chưa có
    if csv_path is None:
        csv_path = os.path.join(log_dir, "metrics.csv")

    if step is not None:
        fieldnames = ["step", "wer"] + list(loss_data.keys())
        rows = []

        # Đọc nội dung hiện có (nếu có)
        if os.path.exists(csv_path):
            with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if int(row["step"]) == step:
                        # Cập nhật dòng hiện tại
                        row = {
                            "step": step,
                            "wer": result["wer"],
                            **loss_data
                        }
                    rows.append(row)
        else:
            # Nếu chưa có file, thêm dòng mới
            rows.append({
                "step": step,
                "wer": result["wer"],
                **loss_data
            })

        # Kiểm tra nếu step chưa có trong file (không bị ghi đè)
        if not any(int(r["step"]) == step for r in rows):
            rows.append({
                "step": step,
                "wer": result["wer"],
                **loss_data
            })

        # Ghi lại toàn bộ file
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return {
        **result,
        "audio": ipd.Audio(audio_output_path, rate=sampling_rate)
    }

def evaluate_all_checkpoints(
    models_dir,
    text_path,
    config_path,
    whisper_model_size="medium"
):
    """
    Chạy đánh giá cho các mô hình G_*.onnx trong thư mục có step chia hết cho 5000.

    Args:
        models_dir (str): Thư mục chứa các file G_*.onnx.
        text_path (str): File văn bản đầu vào.
        config_path (str): File config.
        whisper_model_size (str): Kích thước Whisper model.
    """
    model_files = []
    for f in os.listdir(models_dir):
        match = re.match(r'G_(\d+)\.onnx', f)
        if match:
            step = int(match.group(1))
            if step % 5000 == 0:
                model_files.append((step, os.path.join(models_dir, f)))

    # Sắp xếp theo step tăng dần
    model_files.sort(key=lambda x: x[0])

    print(f"🔍 Phát hiện {len(model_files)} mô hình hợp lệ (chia hết cho 5000) để đánh giá...")

    for step, model_path in model_files:
        print(f"\n🔧 Đánh giá mô hình: {model_path}")
        synthesize_and_evaluate(
            text_path=text_path,
            model_path=model_path,
            config_path=config_path,
            whisper_model_size=whisper_model_size
        )

def plot_metrics(csv_path):
    # Đọc dữ liệu từ CSV
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="step")  # Sắp xếp theo step

    # Tạo biểu đồ
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Trục Y bên trái: các loss
    ax1.set_xlabel('Checkpoint (step)')
    ax1.set_ylabel('Loss values', color='tab:blue')

    loss_keys = ["loss_g_total", "loss_d_total", "loss_mel", "loss_fm", "loss_g_kl"]
    for loss in loss_keys:
        if loss in df.columns:
            ax1.plot(df["step"], df[loss], label=loss)

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Trục Y bên phải: WER
    ax2 = ax1.twinx()
    ax2.set_ylabel('WER (%)', color='tab:red')
    ax2.plot(df["step"], df["wer"] * 100, 'o-', color='tab:red', label='WER')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title('Training Metrics by Checkpoint')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

