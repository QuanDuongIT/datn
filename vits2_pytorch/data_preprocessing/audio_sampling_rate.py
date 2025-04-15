import os
import librosa
import soundfile as sf

def Audio_sampling_rate_folder(target_sr,input_dir,output_dir):
  os.makedirs(output_dir, exist_ok=True)

  for filename in os.listdir(input_dir):
      if filename.endswith(".wav"):
          input_path = os.path.join(input_dir, filename)
          output_path = os.path.join(output_dir, filename)

          # Load file
          y, sr = librosa.load(input_path, sr=None)  # Giữ nguyên SR gốc
          if sr != target_sr:
              # Resample
              y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
          else:
              y_resampled = y

          # Save file
          sf.write(output_path, y_resampled, target_sr)

  print("✅ Chuyển đổi hoàn tất! Các file đã được lưu vào:", output_dir)




"""
chuyển đổi sampling_rate cho jvs_ver1

"""

def Audio_sampling_rate_matrix(root_path,matrix,target_sr):
    rows = len(matrix)
    cols = max(len(row) for row in matrix)

    def Audio_sampling_rate(id_speaker,target_sr):
        input_dir = f"{root_path}jvs_ver1/{id_speaker}/parallel100/wav24kHz16bit"
        output_dir = f"{root_path}jvs_ver1/{id_speaker}/parallel100/wav_22050"
        Audio_sampling_rate_folder(target_sr,input_dir,output_dir)

    for i in range(rows):
        for j in range(len(matrix[i])):
            Audio_sampling_rate(matrix[i][j],target_sr)

