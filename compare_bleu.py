import nltk
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Download tokenizer data

def read_and_process_sequences(file_path, max_length=20):
    """
    Reads sequences from a file, tokenizes them, and ensures they are padded or truncated to a fixed length.
    """
    sequences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Tokenize the line into words
            tokens = word_tokenize(line.strip())
            # Truncate or pad the sequence to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += ["<PAD>"] * (max_length - len(tokens))
            sequences.append(tokens)
    return sequences

def calculate_average_bleu(reference_file, input_file, max_length=20):
    """
    Calculates the average BLEU score between sequences in the reference file and the input file,
    ensuring each sequence is tokenized, padded, or truncated to a fixed length.
    """
    # Read and process sequences
    reference_sequences = read_and_process_sequences(reference_file, max_length)
    input_sequences = read_and_process_sequences(input_file, max_length)
    
    # Compute BLEU scores
    total_bleu = 0
    for inp in input_sequences:
        # BLEU requires references to be a list of lists
        total_bleu += sentence_bleu(reference_sequences, inp)
    
    # Calculate average BLEU
    average_bleu = total_bleu / len(input_sequences)
    return average_bleu

def calculate_max_bleu(reference_file, input_file, max_length=20):
    """
    Calculates the average of max BLEU scores for each generated sequence
    by comparing it to all reference sequences. Ensures each sequence is tokenized,
    padded, or truncated to a fixed length.
    """
    # Read and process sequences
    reference_sequences = read_and_process_sequences(reference_file, max_length)
    input_sequences = read_and_process_sequences(input_file, max_length)
    
    total_max_bleu = 0
    num_generated = len(input_sequences)
    
    for generated in input_sequences:
        # Compute BLEU scores for the generated sequence against all references
        max_bleu = max(
            sentence_bleu([ref], generated) for ref in reference_sequences
        )
        # Accumulate the max BLEU score for this generated sequence
        total_max_bleu += max_bleu
    
    # Return the average of max BLEU scores
    return total_max_bleu / num_generated


def compute_self_bleu_sampled(generated_sequences, sample_size=1000, n_gram=4):
    """
    Computes the average self-BLEU score for a sampled subset of the generated sequences.
    
    Args:
        generated_sequences (list of list of str): Tokenized generated sequences.
        sample_size (int): Number of sequences to sample for self-BLEU calculation.
        n_gram (int): Maximum n-gram size to consider for BLEU.
        
    Returns:
        float: The average self-BLEU score.
    """
    if len(generated_sequences) <= sample_size:
        sample = generated_sequences
    else:
        # Randomly sample sequences without replacement
        sample = random.sample(generated_sequences, sample_size)
    
    total_bleu = 0
    num_sequences = len(sample)
    
    for i, seq in enumerate(sample):
        # All other sequences in the sample are treated as references
        references = [s for j, s in enumerate(sample) if j != i]
        # Compute BLEU for the current sequence
        bleu_score = sentence_bleu(references, seq, weights=(1.0/n_gram,) * n_gram)
        total_bleu += bleu_score
    
    # Return average self-BLEU
    return total_bleu / num_sequences

if __name__ == "__main__":
    # Replace these with the paths to your reference and input files
    reference_file = "mosscap_prompt_injection_sampled_200.txt"
    input_file = "mosscap_epoch0_relstep_injections.txt"
    
    try:
        max_length = 20  # Fixed length for padding/truncation
        
        # Calculate BLEU score against reference
        average_bleu = calculate_average_bleu(reference_file, input_file, max_length)
        print(f"Average BLEU Score: {average_bleu:.4f}")

        max_bleu = calculate_max_bleu(reference_file, input_file, max_length)
        print(f"Max BLEU Score: {max_bleu:.4f}")
        
        # Calculate Self-BLEU for generated sequences
        generated_sequences = read_and_process_sequences(input_file, max_length)
        self_bleu = compute_self_bleu_sampled(generated_sequences)
        print(f"Self-BLEU Score: {self_bleu:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
