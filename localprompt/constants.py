import configparser
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

config = configparser.ConfigParser()
config.read('../config.ini')

INPUT_DIRECTORY = config['DEFAULT']['InputDirectory'].strip('"')
OUTPUT_DIRECTORY = config['DEFAULT']['OutputDirectory'].strip('"')
SENTENCE_MODEL = config['DEFAULT']['SentenceModel'].strip('"')
GPT_MODEL = config['DEFAULT']['GPTModel'].strip('"')
INPUT_FILE = config['DEFAULT']['InputFile'].strip('"')
