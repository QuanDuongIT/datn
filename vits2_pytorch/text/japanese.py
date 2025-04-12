import re
import pyopenjtalk
from unidecode import unidecode

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('％', 'パーセント')
]]

def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_romaji_with_accent(text):
    '''Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html'''
    text = symbols_to_japanese(text)
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = ''
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if text != '':
                text += ' '
            # Sử dụng pyopenjtalk để chuyển văn bản thành phonemes với dấu nhấn
            labels = pyopenjtalk.extract_fullcontext(sentence)
            for n, label in enumerate(labels):
                phoneme = re.search(r'\-([^\+]*)\+', label).group(1)
                if phoneme not in ['sil', 'pau']:  # Loại bỏ các dấu tạm
                    text += phoneme.replace('ch', 'ʧ').replace('sh', 'ʃ').replace('cl', 'Q')
                else:
                    continue
                a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
                a2 = int(re.search(r"\+(\d+)\+", label).group(1))
                a3 = int(re.search(r"\+(\d+)/", label).group(1))
                if re.search(r'\-([^\+]*)\+', labels[n + 1]).group(1) in ['sil', 'pau']:
                    a2_next = -1
                else:
                    a2_next = int(
                        re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
                if a3 == 1 and a2_next == 1:
                    text += ' '
                elif a1 == 0 and a2_next == a2 + 1:
                    text += '↓'
                elif a2 == 1 and a2_next == 2:
                    text += '↑'
        if i < len(marks):
            text += unidecode(marks[i]).replace(' ', '')
    return text


def get_real_sokuon(text):
    """Chuyển các phụ âm đặc biệt như 'sokuon'"""
    # Một số phụ âm sẽ được thay đổi theo quy tắc nhất định (có thể được điều chỉnh thêm tùy yêu cầu)
    return text

def get_real_hatsuon(text):
    """Chuyển các phụ âm đặc biệt như 'hatsuon'"""
    # Cũng có thể chỉnh sửa thêm các quy tắc cho 'hatsuon' nếu cần
    return text


def japanese_to_ipa(text):
    """Chuyển văn bản tiếng Nhật thành IPA"""
    text = japanese_to_romaji_with_accent(text).replace('...', '…')
    text = re.sub(
        r'([aiueo])\1+', lambda x: x.group(0)[0]+'ː'*(len(x.group(0))-1), text)
    text = get_real_sokuon(text)
    text = get_real_hatsuon(text)
    return text


def japanese_to_ipa2(text):
    """Một phiên bản IPA khác của tiếng Nhật (tùy chọn)"""
    text = japanese_to_romaji_with_accent(text).replace('...', '…')
    text = get_real_sokuon(text)
    text = get_real_hatsuon(text)
    return text


def japanese_to_ipa3(text):
    """Một phiên bản IPA khác của tiếng Nhật (tùy chọn)"""
    text = japanese_to_ipa2(text).replace('n^', 'ȵ').replace(
        'ʃ', 'ɕ').replace('*', '\u0325').replace('#', '\u031a')
    text = re.sub(
        r'([aiɯeo])\1+', lambda x: x.group(0)[0]+'ː'*(len(x.group(0))-1), text)
    text = re.sub(r'((?:^|\s)(?:ts|tɕ|[kpt]))', r'\1ʰ', text)
    return text
