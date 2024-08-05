import re


def _is_complete_sentence(line):
    # Use a regular expression to find complete sentences
    # This regex looks for a sentence-ending punctuation followed by a space or end of line
    sentence_endings = re.compile(r'[.!?](?:\s|$)')
    return len(sentence_endings.findall(line)) > 1


def _remove_last_sentence(paragraph):
    # Find all sentence-ending positions
    sentence_endings = [match.start() for match in re.finditer(r'[.!?](?:\s|$)', paragraph)]
    if len(sentence_endings) > 1:
        # Remove the last sentence
        return paragraph[:sentence_endings[-2] + 1].strip()  # Keep until the second-to-last ending
    return paragraph.strip()


def extract_paragraph(lines, start_index, max_size=500):
    """
    Extracts a paragraph containing the specified start index.

    Args:
        lines (list of str): List of text lines from the file.
        start_index (int): The index where the keyword was found.
        max_size (int): Maximum size of the paragraph in characters.

    Returns:
        str: Extracted paragraph or nearby lines if paragraph is too long.
    """

    # Check if the line contains multiple complete sentences
    if _is_complete_sentence(lines[start_index].strip()):
        paragraph = lines[start_index].strip()
        if len(paragraph) <= max_size:
            return paragraph
        else:
            # Try removing the last sentence if the paragraph is too large
            while len(paragraph) > max_size:
                paragraph = _remove_last_sentence(paragraph)
            return paragraph

    # Initialize paragraph extraction
    start = start_index
    end = start_index

    # Extend backwards to find the start of the paragraph
    while start > 0 and lines[start - 1].strip() != "":
        start -= 1

    # Extend forwards to find the end of the paragraph
    while end < len(lines) - 1 and lines[end + 1].strip() != "":
        end += 1

    # Combine lines into a paragraph
    paragraph = "\n".join(lines[start:end + 1]).strip()

    # Check if the paragraph is within the maximum size
    if len(paragraph) <= max_size:
        return paragraph
    else:
        # Try removing the last sentence if the paragraph is too large
        while len(paragraph) > max_size:
            paragraph = _remove_last_sentence(paragraph)
        return paragraph
