import tkinter as tk
from tkinter import scrolledtext

from query_and_answer import *


class ChatApp:
    def __init__(self, root, response_function):
        self.root = root
        self.response_function = response_function  # Store the external response function
        self.root.title("LocalPrompt")

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', height=15)
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Frame for user input and button
        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        # User input area
        self.user_input = tk.Entry(input_frame, width=80)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)  # Bind Enter key to send message

        # Send button
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)

    def send_message(self, event=None):
        # Get user input
        user_message = self.user_input.get().strip()
        if user_message:
            self.display_message("User", user_message)
            self.user_input.delete(0, tk.END)  # Clear input field

            # Generate and display response using the external function
            response = self.response_function(user_message)
            self.display_message("Bot", response)

    def display_message(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.yview(tk.END)  # Scroll to the end
        self.chat_display.config(state='disabled')


def main():
    # Load the Model
    gpt_tokenizer, gpt_model, gpt_device = load_gpt_neo()

    # Load the sentence transformer model
    sentence_model = SentenceTransformer(SENTENCE_MODEL)

    def my_response_function(query):
        # Load and query embeddings
        similar_chunks = load_and_query_embeddings(query, sentence_model, OUTPUT_DIRECTORY)

        # Use the most similar chunks as context
        context = " ".join(similar_chunks)

        # Generate answer
        return generate_answer(query, context, gpt_model, gpt_tokenizer, gpt_device)

    root = tk.Tk()
    app = ChatApp(root, my_response_function)
    root.geometry("600x400")
    root.mainloop()


if __name__ == "__main__":
    main()
