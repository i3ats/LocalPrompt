import os
import pickle

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from constants import OUTPUT_DIRECTORY, SENTENCE_MODEL


# Load GPT-2 Medium model and tokenizer
def load_gpt2_medium():
    print("Loading GPT-2 Medium...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", pad_token='<|endoftext|>')
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# Load embeddings and perform queries
def load_and_query_embeddings(query, sentence_model, output_dir="vector_db", top_k=5):
    """Loads embeddings and performs a similarity search."""
    # Load the Faiss index
    index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))

    # Load metadata
    with open(os.path.join(output_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    chunks = metadata["chunks"]

    # Generate the query embedding using the sentence transformer
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().numpy().astype('float32')

    # Perform the search
    _, indices = index.search(query_embedding, top_k)
    similar_chunks = [chunks[i] for i in indices[0]]

    return similar_chunks

# Generate an answer using GPT-2
def generate_answer(query, context, model, tokenizer, device, max_new_tokens=150):
    # Calculate max input length allowed by GPT-2
    # max_input_length = 1024 - max_new_tokens
    # TODO figure out what this value should be
    max_input_length = 924 - max_new_tokens

    # Tokenize the query
    query_tokens = tokenizer.encode(query, add_special_tokens=False)

    # Tokenize the context and truncate if necessary
    context_tokens = tokenizer.encode(context, add_special_tokens=False)

    # Adjust context length if necessary to fit within max input length
    if len(context_tokens) + len(query_tokens) + 3 > max_input_length:
        # Truncate context
        context_tokens = context_tokens[:max_input_length - len(query_tokens) - 3]

    # Prepare the input prompt
    prompt = tokenizer.decode(context_tokens) + f"\n\nQuestion: {query}\n\nAnswer:"

    # Tokenize and prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

    # Generate text with attention mask
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Main function
def main():
    # Load the GPT-2 Medium model
    gpt_tokenizer, gpt_model, gpt_device = load_gpt2_medium()

    # Load the sentence transformer model
    sentence_model = SentenceTransformer(SENTENCE_MODEL)

    # Define a query
    query = "Who is Mira?"

    # Load and query embeddings
    similar_chunks = load_and_query_embeddings(query, sentence_model, OUTPUT_DIRECTORY)

    # Use the most similar chunks as context
    context = " ".join(similar_chunks)

    # Generate answer
    answer = generate_answer(query, context, gpt_model, gpt_tokenizer, gpt_device)
    print("Generated Answer:", answer)

if __name__ == "__main__":
    main()
