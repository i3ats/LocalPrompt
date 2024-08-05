import pdfplumber


def extract_text_from_pdf(pdf_path, output_file_path):
    """
    Extracts text from a PDF file and saves it to a new text file.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_file_path (str): Path to the output text file.
    """
    try:
        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            # Open the output text file in write mode
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                # Iterate over each page in the PDF
                for page in pdf.pages:
                    # Extract text from the page
                    text = page.extract_text()
                    if text:
                        # Write the extracted text to the output file
                        output_file.write(text + '\n')
        print(f"Text extracted and saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
pdf_path = "C:\\Users\\joe_v\\OneDrive\\Desktop\\Knowledgebase\\Starfinder - Core Rulebook.pdf"  # Replace with your PDF file path
output_file_path = "C:\\Users\\joe_v\\OneDrive\\Desktop\\Knowledgebase\\Starfinder - Core Rulebook.txt"  # Replace with desired output text file path

extract_text_from_pdf(pdf_path, output_file_path)
