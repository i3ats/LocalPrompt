import logging

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from constants import INPUT_FILE
from extract_keywords import extract_keywords
from extract_paragraph import extract_paragraph
from file_functions import detect_file_encoding


def search_file_for_keywords(file_path, keywords, max_paragraph_size=500):
    """
    Searches a text file for given keywords and extracts sections containing any of the keywords.

    Args:
        file_path (str): Path to the text file.
        keywords (list of str): The list of keywords to search for.
        max_paragraph_size (int): Maximum size of a paragraph to extract in characters.

    Returns:
        list of str: A list of extracted text sections containing any of the keywords.
    """
    encoding = detect_file_encoding(file_path)

    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    keyword_sections = []
    logging.info(f"Searching for keywords {keywords} in the document...")
    for i, line in enumerate(lines):
        if any(keyword.lower() in line.lower() for keyword in keywords):
            # Attempt to extract a full paragraph
            section = extract_paragraph(lines, i, max_paragraph_size)
            keyword_sections.append(section)
            logging.debug(f"Keyword found at line {i}: {section[:75]}...")  # Log the first 75 characters

    logging.info(f"Total sections found with keywords {keywords}: {len(keyword_sections)}")
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
    logging.info("Ranking sections by relevance...")
    if not sections:
        logging.warning("No sections available to rank.")
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sections)
    keyword_vector = vectorizer.transform([" ".join(keywords)])

    # Calculate relevance scores
    scores = (tfidf_matrix @ keyword_vector.T).toarray().flatten()

    # Sort sections by score
    ranked_sections = [section for _, section in sorted(zip(scores, sections), reverse=True)]
    logging.info("Sections ranked.")
    return ranked_sections


def summarize_combined_sections(sections, keywords, top_n=3):
    """
    Summarizes the combined text of the top N most relevant sections using DistilBART.

    Args:
        sections (list of str): The list of text sections to summarize.
        keywords (list of str): The list of keywords to emphasize in the summary.
        top_n (int): Number of top sections to combine for the summary.

    Returns:
        str: A summary of the combined text.
    """
    if not sections:
        logging.warning("No sections available to summarize.")
        return ""

    # Determine the device to use
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    logging.info("Loading DistilBART model and tokenizer locally...")
    # Load the DistilBART model and tokenizer locally
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    logging.info(f"Combining the top {top_n} sections for summarization...")
    # Combine the top N sections into a single text
    combined_text = " ".join(sections[:top_n])

    # Prepare a prompt including keywords to guide the summarization
    keyword_prompt = "Keywords: " + ", ".join(keywords) + "\n"

    # If the combined text is too long, break it down into chunks to summarize in parts
    max_chunk_length = 512  # Reduce chunk size for more focused input
    overlap = 128  # Introduce overlap for consistent context
    chunks = [
        combined_text[i:i + max_chunk_length]
        for i in range(0, len(combined_text), max_chunk_length - overlap)
    ]

    logging.info(f"Summarizing {len(chunks)} chunks...")
    # Generate a summary for each chunk using DistilBART
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logging.debug(f"Summarizing chunk {i + 1}...")
        input_text = keyword_prompt + chunk
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = model.generate(inputs["input_ids"], max_length=256, min_length=100, num_beams=3,
                                     no_repeat_ngram_size=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(summary)

    # Combine the summaries of the chunks to create a final summary
    final_summary = " ".join(chunk_summaries)

    # Ensure the final summary does not exceed the max desired length
    if len(tokenizer.encode(final_summary)) > 1024:
        final_summary = tokenizer.decode(tokenizer.encode(final_summary)[:1024], skip_special_tokens=True)

    logging.info("Summarization complete.")
    return final_summary


# Example usage
file_path = INPUT_FILE
# prompt = "What is an Solarian?"
prompt = "What are the Shadowfell Rings?"
keywords = extract_keywords(prompt)

if keywords:
    extracted_sections = search_file_for_keywords(file_path, keywords)
    logging.info(f"\nSections containing the extracted keywords {keywords}:\n")
    for section in extracted_sections:
        logging.debug(section)
        logging.info("-" * 40)

    # Rank sections by relevance
    ranked_sections = rank_sections_by_relevance(extracted_sections, keywords)

    # Summarize the combined information from the most relevant sections
    combined_summary = summarize_combined_sections(ranked_sections, keywords)
    logging.info("\nCombined Summary of the most relevant extracted sections:\n")
    logging.info(combined_summary)
else:
    logging.warning("No keywords extracted. Please provide a more detailed prompt.")
