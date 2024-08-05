import chardet
from transformers import pipeline


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
    return encoding


def search_file_for_keyword(file_path, keyword, context_lines=2):
    """
    Searches a text file for a given keyword and extracts sections containing the keyword.

    Args:
        file_path (str): Path to the text file.
        keyword (str): The keyword to search for.
        context_lines (int): Number of lines to include before and after the keyword occurrence.

    Returns:
        list of str: A list of extracted text sections containing the keyword.
    """
    encoding = detect_file_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    keyword_sections = []
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            # Get context lines before and after the keyword occurrence
            start = max(i - context_lines, 0)
            end = min(i + context_lines + 1, len(lines))
            section = "".join(lines[start:end])
            keyword_sections.append(section)

    return keyword_sections


def summarize_combined_sections(sections):
    """
    Summarizes the combined text of all sections using GPT-Neo.

    Args:
        sections (list of str): The list of text sections to summarize.

    Returns:
        str: A summary of the combined text.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    # Combine all sections into a single text
    combined_text = " ".join(sections)

    # If the combined text is too long, break it down into chunks to summarize in parts
    max_chunk_length = 1024  # Maximum number of tokens for the model to handle effectively
    chunks = [combined_text[i:i + max_chunk_length] for i in range(0, len(combined_text), max_chunk_length)]

    # Generate a summary for each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        chunk_summaries.append(summary)

    # Combine the summaries of the chunks to create a final summary
    final_summary = " ".join(chunk_summaries)

    return final_summary


# Example usage
file_path = "C:\\Users\\joe_v\\OneDrive\\Desktop\\Guild\\guild_book_text.txt"
keyword = 'Ironbound'
extracted_sections = search_file_for_keyword(file_path, keyword)

print(f"Sections containing the keyword '{keyword}':\n")
for section in extracted_sections:
    print(section)
    print("-" * 40)

# Summarize the combined information from all sections
combined_summary = summarize_combined_sections(extracted_sections)

print("Combined Summary of the extracted sections:\n")
print(combined_summary)
