# test_math_operations.py

import unittest

from src.extract_keywords import extract_keywords


class TestMathOperations(unittest.TestCase):

    def test_easy(self):
        prompt = "Mira"
        self.assertEqual(['Mira'], extract_keywords(prompt))

    def test_simple_question(self):
        prompt = "Who is Mira?"
        self.assertEqual(['Mira'], extract_keywords(prompt))

    def test_simple_request(self):
        prompt = "Describe Mira"
        self.assertEqual(['Describe', 'Mira'], extract_keywords(prompt))

    def test_longer_request(self):
        prompt = "tell me about Mira"
        self.assertEqual(['Mira'], extract_keywords(prompt))

    def test_statement(self):
        prompt = "Analyze the influence of the Ironbound"
        self.assertEqual(['Analyze', 'Ironbound'], extract_keywords(prompt))


if __name__ == '__main__':
    unittest.main()
