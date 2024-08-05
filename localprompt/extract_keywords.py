from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def extract_keywords(prompt, num_keywords=5):
    """
    Extracts relevant keywords from a prompt string using NLTK.

    Args:
        prompt (str): The input prompt from which to extract keywords.
        num_keywords (int): The number of keywords to extract.

    Returns:
        list of str: A list of extracted keywords.
    """
    print("Extracting keywords from the prompt...")

    # Define stop words and common question words
    stop_words = set(stopwords.words('english'))
    question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were'}

    # Tokenize the prompt
    words = word_tokenize(prompt)

    # Retain words that are capitalized and not common question words
    filtered_words = [
        word for word in words
        if word.isalpha() and
           (word.lower() not in stop_words or word[0].isupper()) and
           word.lower() not in question_words
    ]

    # Extract proper nouns and capitalized words
    proper_nouns = [word for word in filtered_words if word[0].isupper()]

    # Handle case where no proper nouns are found
    if not proper_nouns:
        print("No proper nouns found in the prompt after filtering.")
        return []

    # Return top proper nouns as keywords
    keywords = proper_nouns[:num_keywords]
    print(f"Extracted keywords: {keywords}")
    return keywords
