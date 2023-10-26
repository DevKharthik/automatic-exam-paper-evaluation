import spacy

# Load the English language model in spaCy
nlp = spacy.load("en_core_web_sm")

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to calculate semantic similarity using spaCy
def calculate_similarity(original_text, user_text):
    doc1 = nlp(original_text)
    doc2 = nlp(user_text)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

# Path to the original answer text file
original_answer_text_path = r"C:\paper project\HTRPipeline-master\scripts\answer.txt"

# Path to the user's answer text file
user_answer_text_path = r"C:\paper project\HTRPipeline-master\scripts\sentence.txt"

# Read text from the files
original_answer_text = read_text_from_file(original_answer_text_path)
user_answer_text = read_text_from_file(user_answer_text_path)

# Calculate NLP-based similarity score (adjust weight as needed)
nlp_similarity_weight = 1.0  # Use 1.0 to consider only NLP-based similarity
nlp_similarity_score = calculate_similarity(original_answer_text, user_answer_text)

# Set a threshold for the NLP-based similarity score (adjust as needed)
nlp_similarity_threshold = 0.85

# Evaluate correctness based on the NLP-based similarity score
# if nlp_similarity_score >= nlp_similarity_threshold:
#     print("The user's answer is correct.")
# else:
#     print("The user's answer is incorrect.")

# Calculate the final percentage based on the NLP-based similarity
final_percentage = nlp_similarity_score * 100

print(f"NLP-Based Similarity Score: {nlp_similarity_score:.2f}")
print(f"Final Percentage: {final_percentage:.2f}%")