import os
import pickle

import chardet
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from constants import *


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PDFPlumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle cases where extract_text might return None
    return text

# Detect file encoding
def detect_file_encoding(file_path, num_bytes=10000):
    """Detects the encoding of a file by reading a specified number of bytes."""
    with open(file_path, 'rb') as file:
        raw_data = file.read(num_bytes)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"Detected encoding: {encoding} with confidence {confidence}")
    return encoding

# Extract text from .txt file
def extract_text_from_txt(file_path):
    """Extracts text from a .txt file with encoding detection."""
    # Detect the encoding of the file
    encoding = detect_file_encoding(file_path)

    # Read the file using the detected encoding
    with open(file_path, 'r', encoding=encoding) as file:
        text = file.read()
    return text

def split_text_into_chunks(text, max_chunk_size=150):
    """Splits text into chunks of a specified maximum size."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + len(word) + 1 > max_chunk_size:
            if current_chunk:  # Ensure the chunk is not empty before adding
                chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def generate_embeddings(text_chunks, model):
    """Generates embeddings for a list of text chunks."""
    embeddings = model.encode(text_chunks, convert_to_tensor=True, show_progress_bar=True)

    # Debugging: Check embedding shape
    if len(embeddings) == 0:
        raise ValueError("No embeddings were generated. Check your input data.")

    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings


def store_embeddings(embeddings, chunks, output_dir="vector_db"):
    """Stores embeddings and corresponding text chunks in a common vector database."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert embeddings to a numpy array
    embeddings_np = np.array(embeddings.cpu()).astype('float32')

    # Check if embeddings are non-empty and have the correct shape
    if embeddings_np.size == 0 or embeddings_np.ndim != 2:
        raise ValueError("Invalid embeddings data. Ensure embeddings are generated correctly.")

    # Initialize or load an existing Faiss index
    index_file = os.path.join(output_dir, "faiss_index.bin")
    metadata_file = os.path.join(output_dir, "metadata.pkl")

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            all_chunks = metadata['chunks']
    else:
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        all_chunks = []

    # Add new embeddings to the index
    index.add(embeddings_np)
    all_chunks.extend(chunks)

    # Debugging: Check index size
    print(f"Index size after adding embeddings: {index.ntotal}")

    # Save the updated index and metadata
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump({"chunks": all_chunks}, f)


def process_directory(directory, output_dir="vector_db"):
    """Processes all .pdf and .txt files in the directory and stores their embeddings in a common index."""
    sentence_model = SentenceTransformer(SENTENCE_MODEL)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.endswith('.pdf'):
            print(f"Processing PDF file: {filename}")
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            print(f"Processing text file: {filename}")
            text = extract_text_from_txt(file_path)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        if text.strip():  # Ensure text is not empty
            chunks = split_text_into_chunks(text)
            embeddings = generate_embeddings(chunks, sentence_model)
            store_embeddings(embeddings, chunks, output_dir)
        else:
            print(f"No text found in {filename}. Skipping.")


# Main function
def main():
    # Specify the directory containing the files
    directory = INPUT_DIRECTORY

    # Output directory for the vector database
    output_dir = OUTPUT_DIRECTORY

    # Process all files in the directory
    process_directory(directory, output_dir)

if __name__ == "__main__":
    main()
