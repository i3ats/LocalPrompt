import chardet
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

from extract_keywords import extract_keywords


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
    print(f"Detected file encoding: {encoding}")
    return encoding


def search_file_for_keywords(file_path, keywords, context_before=2, context_after=2):
    """
    Searches a text file for given keywords and extracts sections containing any of the keywords.

    Args:
        file_path (str): Path to the text file.
        keywords (list of str): The list of keywords to search for.
        context_before (int): Number of lines to include before the keyword occurrence.
        context_after (int): Number of lines to include after the keyword occurrence.

    Returns:
        list of str: A list of extracted text sections containing any of the keywords.
    """
    encoding = detect_file_encoding(file_path)

    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    keyword_sections = []
    print(f"Searching for keywords {keywords} in the document...")
    for i, line in enumerate(lines):
        if any(keyword.lower() in line.lower() for keyword in keywords):
            # Get context lines before and after the keyword occurrence
            start = max(i - context_before, 0)
            end = min(i + context_after + 1, len(lines))
            section = "".join(lines[start:end])
            keyword_sections.append(section)
            print(f"Keyword found at line {i}: {section[:75]}...")  # Print the first 75 characters

    print(f"Total sections found with keywords {keywords}: {len(keyword_sections)}")
    return keyword_sections

def rank_sections_by_relevance(sections, keywords):
    """
    Ranks sections based on their relevance to the list of keywords using TF-IDF.

    Args:
        sections (list of str): The list of text sections to rank.
        keywords (list of str): The list of keywords to determine relevance.

    Returns:
        list of str: A sorted list of sections by relevance.
    """
    print("Ranking sections by relevance...")
    if not sections:
        print("No sections available to rank.")
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sections)
    keyword_vector = vectorizer.transform([" ".join(keywords)])

    # Calculate relevance scores
    scores = (tfidf_matrix @ keyword_vector.T).toarray().flatten()

    # Sort sections by score
    ranked_sections = [section for _, section in sorted(zip(scores, sections), reverse=True)]
    print("Sections ranked.")
    return ranked_sections

def summarize_combined_sections(sections, top_n=3):
    """
    Summarizes the combined text of the top N most relevant sections using BART.

    Args:
        sections (list of str): The list of text sections to summarize.
        top_n (int): Number of top sections to combine for the summary.

    Returns:
        str: A summary of the combined text.
    """
    if not sections:
        print("No sections available to summarize.")
        return ""

    # Determine the device to use
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    print(f"Initializing BART summarization pipeline...")
    # Initialize the BART summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    print(f"Combining the top {top_n} sections for summarization...")
    # Combine the top N sections into a single text
    combined_text = " ".join(sections[:top_n])

    # If the combined text is too long, break it down into chunks to summarize in parts
    max_chunk_length = 1024  # Maximum number of tokens for the model to handle effectively
    chunks = [combined_text[i:i + max_chunk_length] for i in range(0, len(combined_text), max_chunk_length)]

    print(f"Summarizing {len(chunks)} chunks...")
    # Generate a summary for each chunk using BART
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}...")
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        chunk_summaries.append(summary)

    # Combine the summaries of the chunks to create a final summary
    final_summary = " ".join(chunk_summaries)

    print("Summarization complete.")
    return final_summary

# Example usage
file_path = "C:\\Users\\joe_v\\OneDrive\\Desktop\\Guild\\guild_book_text.txt"
prompt = "Who is Ashfeather?"
keywords = extract_keywords(prompt)

if keywords:
    extracted_sections = search_file_for_keywords(file_path, keywords, context_before=2, context_after=3)
    print(f"\nSections containing the extracted keywords {keywords}:\n")
    for section in extracted_sections:
        print(section)
        print("-" * 40)

    # Rank sections by relevance
    ranked_sections = rank_sections_by_relevance(extracted_sections, keywords)

    # Summarize the combined information from the most relevant sections
    combined_summary = summarize_combined_sections(ranked_sections)
    print("\nCombined Summary of the most relevant extracted sections:\n")
    print(combined_summary)
else:
    print("No keywords extracted. Please provide a more detailed prompt.")
