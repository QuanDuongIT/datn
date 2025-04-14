
# # dataset: jsut_ver1.1: 

# def replace_colon_with_pipe(input_path, output_path):
#     """
#     Thay thế dấu ':' thành '|' trong file văn bản và ghi ra file mới.
    
#     Parameters:
#     - input_path (str): đường dẫn tới file gốc.
#     - output_path (str): đường dẫn tới file sau khi sửa.
#     """
#     try:
#         with open(input_path, "r", encoding="utf-8") as f:
#             content = f.read()

#         # Thay thế
#         content_modified = content.replace(":", "|")

#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(content_modified)

#         print(f"✅ File đã được sửa và lưu thành: {output_path}")
    
#     except FileNotFoundError:
#         print(f"❌ File không tồn tại: {input_path}")
#     except Exception as e:
#         print(f"⚠️ Lỗi xảy ra: {e}")

# # replace_colon_with_pipe('jsut_ver1.1/repeat500/transcript_utf8.txt', 'jsut_ver1.1/repeat500/transcript_modified.txt')

# # folders = ['basic5000', 'precedent130','repeat500','voiceactress100','loanword128','travel1000','countersuffix26','onomatopee300','utparaphrase512']

