""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols
from text import japanese 

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        if symbol in _symbol_to_id.keys():
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            continue
    return sequence


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    
    for symbol in cleaned_text:
        if symbol in _symbol_to_id.keys():
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            continue
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        # Kiểm tra và sử dụng bộ làm sạch từ japanese.py nếu tên bộ làm sạch khớp
        if name == 'japanese_to_ipa2':
            text = japanese.japanese_to_ipa2(text)  # Sử dụng hàm japanese_to_ipa2 trong japanese.py
        elif name == 'japanese_to_ipa':
            text = japanese.japanese_to_ipa(text)  # Sử dụng hàm japanese_to_ipa trong japanese.py
        elif name == 'japanese_to_ipa3':
            text = japanese.japanese_to_ipa3(text)  # Sử dụng hàm japanese_to_ipa3 trong japanese.py
        else:
            # Đối với các bộ làm sạch khác, sử dụng cleaners từ thư mục cleaners
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception(f"Unknown cleaner: {name}")
            text = cleaner(text)
    return text
