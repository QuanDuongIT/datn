import random
import os

audionamesid = 'ja_audio_sid_text'

def process_sid_and_split_text(inpath, outdir, seed=42):
    os.makedirs(outdir, exist_ok=True)

    valid_lines = []

    with open(inpath, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            original_line = line.strip()

            # Bỏ qua dòng trống hoặc không có đúng 2 dấu "|"
            if not original_line or original_line.count('|') != 2:
                print(f"❌ Dòng {line_num} bị bỏ qua (sai định dạng '|'): {original_line}")
                continue

            parts = original_line.split('|')
            audio_path, sid, text = parts

            if not audio_path.strip() or not sid.strip() or not text.strip():
                print(f"❌ Dòng {line_num} bị bỏ qua (trường thiếu dữ liệu): {original_line}")
                continue

            # Chuyển đổi "jsut_ver1.1" -> "DUMMY2"
            audio_path = audio_path.replace("jvs_ver1", "DUMMY2")

            # Thêm ".wav" nếu chưa có
            if not os.path.basename(audio_path).endswith(".wav"):
                audio_path += ".wav"

            # Kiểm tra đường dẫn tồn tại
            full_audio_path = os.path.join('/content/datn/vits2_pytorch', audio_path)
            if not os.path.exists(full_audio_path):
                print(f"⚠️ Dòng {line_num} bỏ qua (file không tồn tại): {full_audio_path}")
                continue

            # Hợp lệ
            valid_lines.append(f"{audio_path}|{sid}|{text}")

    total = len(valid_lines)
    if total == 0:
        print("❌ Không có dòng hợp lệ nào để xử lý.")
        return

    # Shuffle và chia tập
    random.seed(seed)
    random.shuffle(valid_lines)

    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.15)

    train_lines = valid_lines[:train_end]
    val_lines = valid_lines[train_end:val_end]
    test_lines = valid_lines[val_end:]

    def write_list(lines, split_name):
        out_path = os.path.join(outdir, f"{audionamesid}_{split_name}_filelist.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    write_list(train_lines, "train")
    write_list(val_lines, "val")
    write_list(test_lines, "test")

    print(f"\n✅ Đã xử lý {total} dòng hợp lệ:")
    print(f" - Train: {len(train_lines)} dòng")
    print(f" - Val:   {len(val_lines)} dòng")
    print(f" - Test:  {len(test_lines)} dòng")



audioname = 'ja_audio_text'

def process_and_split_text(inpath, outdir, seed=42):
    os.makedirs(outdir, exist_ok=True)

    valid_lines = []
    with open(inpath, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or '|' not in line:
                continue  # Bỏ qua dòng rỗng hoặc không có '|'
            parts = line.split('|')
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue  # Bỏ qua dòng thiếu thông tin file hoặc text
            # Đổi đường dẫn đầu vào
            line = line.replace("isla/", "DUMMY1/")
            valid_lines.append(line)

    total = len(valid_lines)
    if total == 0:
        print("⚠️ Không có dòng hợp lệ nào để xử lý.")
        return

    random.seed(seed)
    random.shuffle(valid_lines)

    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.15)

    train_lines = valid_lines[:train_end]
    val_lines = valid_lines[train_end:val_end]
    test_lines = valid_lines[val_end:]

    with open(os.path.join(outdir, f"{audioname}_train_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(outdir, f"{audioname}_val_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(os.path.join(outdir, f"{audioname}_test_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"✅ Đã xử lý {total} dòng hợp lệ:")
    print(f" - Train: {len(train_lines)} dòng")
    print(f" - Val:   {len(val_lines)} dòng")
    print(f" - Test:  {len(test_lines)} dòng")

def process_and_split_text_jsut(inpath, outdir, seed=42):
    os.makedirs(outdir, exist_ok=True)

    valid_lines = []
    with open(inpath, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or '|' not in line:
                continue  # Bỏ qua dòng rỗng hoặc không có '|'
            parts = line.split('|')
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue  # Bỏ qua dòng thiếu thông tin file hoặc text
            # Đổi đường dẫn đầu vào
            line = line.replace("jsut_ver/", "DUMMY1/")
            line = line.replace("/wav", "/wav_22050")
            valid_lines.append(line)

    total = len(valid_lines)
    if total == 0:
        print("⚠️ Không có dòng hợp lệ nào để xử lý.")
        return

    random.seed(seed)
    random.shuffle(valid_lines)

    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.199)

    train_lines = valid_lines[:train_end]
    val_lines = valid_lines[train_end:val_end]
    test_lines = valid_lines[val_end:]

    with open(os.path.join(outdir, f"{audioname}_train_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(outdir, f"{audioname}_val_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(os.path.join(outdir, f"{audioname}_test_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"✅ Đã xử lý {total} dòng hợp lệ:")
    print(f" - Train: {len(train_lines)} dòng")
    print(f" - Val:   {len(val_lines)} dòng")
    print(f" - Test:  {len(test_lines)} dòng")

def process_and_split_text_css10_japanese(inpath, outdir, seed=42):
    os.makedirs(outdir, exist_ok=True)

    valid_lines = []
    with open(inpath, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or '|' not in line:
                continue  # Bỏ qua dòng rỗng hoặc không có '|'
            parts = line.split('|')
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue  # Bỏ qua dòng thiếu thông tin file hoặc text
            # Đổi đường dẫn đầu vào
            line = line.replace("meian/", "DUMMY1/")
            valid_lines.append(line)

    total = len(valid_lines)
    if total == 0:
        print("⚠️ Không có dòng hợp lệ nào để xử lý.")
        return

    random.seed(seed)
    random.shuffle(valid_lines)

    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.199)

    train_lines = valid_lines[:train_end]
    val_lines = valid_lines[train_end:val_end]
    test_lines = valid_lines[val_end:]

    with open(os.path.join(outdir, f"{audioname}_train_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(outdir, f"{audioname}_val_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(os.path.join(outdir, f"{audioname}_test_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"✅ Đã xử lý {total} dòng hợp lệ:")
    print(f" - Train: {len(train_lines)} dòng")
    print(f" - Val:   {len(val_lines)} dòng")
    print(f" - Test:  {len(test_lines)} dòng")