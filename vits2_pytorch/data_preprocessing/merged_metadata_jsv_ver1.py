import os

def merged_metadata_jsv_ver1_parallel100(root_path,matrix):
    def convert_transcript_to_metadata(id_speacker, sid_id, pep_filename):
        input_path = f'{root_path}jvs_ver1/{id_speacker}/parallel100/transcripts_utf8.txt'
        output_path = f'{root_path}jvs_ver1/{id_speacker}/parallel100/metadata.txt'
        with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.replace(':', f'.wav|{sid_id}|', 1)
                line = line.replace(f'{pep_filename}', f'jvs_ver1/{id_speacker}/parallel100/wav_22050/{pep_filename}')
                outfile.write(line)
        return output_path  # Trả về đường dẫn để dùng sau

    # Danh sách chứa đường dẫn các metadata.txt để gộp
    metadata_paths = []

    # Giả sử matrix đã được định nghĩa từ trước
    rows = len(matrix)
    cols = max(len(row) for row in matrix)

    for i in range(rows):
        for j in range(len(matrix[i])):
            speaker_id = f'{matrix[i][j]}'
            sid_id = f'{i}'
            metadata_path = convert_transcript_to_metadata(speaker_id, sid_id, 'VOICEACTRESS')
            metadata_paths.append(metadata_path)

    # Gộp tất cả metadata.txt vào một file duy nhất
    merged_output_path = f'{root_path}jvs_ver1/merged_metadata.txt'
    with open(merged_output_path, 'w', encoding='utf-8') as merged_file:
        for path in metadata_paths:
            with open(path, 'r', encoding='utf-8') as single_file:
                merged_file.write(single_file.read())

    print(f"Đã gộp tất cả metadata.txt vào {merged_output_path}")
