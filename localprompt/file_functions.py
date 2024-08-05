import logging

import chardet


def detect_file_encoding(file_path):
    """
    Detects the encoding of a text file using chardet.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Detected file encoding.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        logging.info(f"Detected file encoding: {encoding}")
        return encoding
