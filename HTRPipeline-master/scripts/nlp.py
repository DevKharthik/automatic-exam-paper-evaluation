import nltk
nltk.download('punkt')

from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize

with open(r'C:\paper project\HTRPipeline-master\scripts\sentence.txt', 'r') as file:
    text = file.read()

text = text.upper()

words = word_tokenize(text)

spell = SpellChecker()

corrected_text = []
for word in words:
    corrected_word = spell.correction(word)
    corrected_text.append(corrected_word)

corrected_text = ' '.join(corrected_text)

new_text=corrected_text.upper()

with open(r'C:\paper project\HTRPipeline-master\scripts\sentence.txt', 'w') as file:
    file.write(new_text)