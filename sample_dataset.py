from datasets import load_dataset
import random
from difflib import SequenceMatcher
import nltk

# Ensure NLTK tokenizer is available
nltk.download("punkt")
from nltk.tokenize import word_tokenize

# Load the dataset
dataset = load_dataset("Lakera/mosscap_prompt_injection")

# Function to check if two sequences are similar based on a threshold
def is_similar(seq1, seq2, threshold=0.8):
    similarity = SequenceMatcher(None, seq1, seq2).ratio()
    return similarity >= threshold

# Function for stratified sampling with no duplicates or similar sequences
def stratified_sample_unique_and_dissimilar(dataset, size, similarity_threshold=0.8):
    levels = list(set(dataset["level"]))  # Get unique levels
    per_level = size // len(levels)       # Number of samples per level
    sampled_prompts = []
    
    for level in levels:
        # Filter by level
        level_data = dataset.filter(lambda x: x["level"] == level)
        shuffled = level_data.shuffle(seed=42)  # Shuffle the data
        count = 0
        
        for item in shuffled:
            # Tokenize and truncate prompt to 20 tokens
            tokenized_prompt = word_tokenize(item["prompt"])
            truncated_prompt = " ".join(tokenized_prompt[:20])
            
            # Check if it's similar to existing prompts
            if all(not is_similar(truncated_prompt, existing, similarity_threshold) for existing in sampled_prompts):
                sampled_prompts.append(truncated_prompt)
                count += 1
                if count >= per_level:
                    break
    
    random.shuffle(sampled_prompts)  # Shuffle the final combined dataset
    return sampled_prompts[:size]  # Ensure total size matches the target

# Save sampled datasets as text files
def save_as_text(data, file_path):
    with open(file_path, "w") as f:
        for sequence in data:
            f.write(sequence + "\n")  # Write each truncated prompt as a single line

# Generate stratified samples with unique and dissimilar sequences
sampled_1000 = stratified_sample_unique_and_dissimilar(dataset["train"], 1000)
sampled_200 = stratified_sample_unique_and_dissimilar(dataset["train"], 200)

# Save the datasets
save_as_text(sampled_1000, "mosscap_prompt_injection_sampled_1000.txt")
save_as_text(sampled_200, "mosscap_prompt_injection_sampled_200.txt")