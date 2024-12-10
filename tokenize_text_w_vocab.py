import nltk
import pickle

# Note that I made my padding token be len(vocab) = 1965

def validate_tokens(data_file, vocab_size):
    with open(data_file, 'r') as f:
        for line in f:
            tokens = map(int, line.strip().split())
            for token in tokens:
                if token < 0 or token > vocab_size:
                    print(token)
                    raise ValueError(f"Token {token} out of bounds for vocab_size {vocab_size}")


def tokenize_and_pad(file, word_to_index, seq_len):
    """
    Tokenizes and pads the sequences in the file using the provided vocabulary and sequence length.
    """
    tokenized_sequences = []

    with open(file, 'r', encoding='utf-8') as raw:
        for line in raw:
            # Tokenize the line
            tokens = nltk.word_tokenize(line.lower())
            
            # Convert tokens to indices using the word_to_index dictionary
            token_indices = [int(word_to_index.get(token, len(word_to_index))) for token in tokens]

            # Pad or truncate the sequence to match seq_len
            if len(token_indices) < seq_len:
                token_indices += [len(word_to_index)] * (seq_len - len(token_indices))  # EOF padding
            else:
                token_indices = token_indices[:seq_len]  # Truncate if longer than seq_len

            tokenized_sequences.append(token_indices)

    return tokenized_sequences


def save_tokenized_data(tokenized_sequences, output_file):
    """
    Saves the tokenized sequences to a file, one sequence per line.
    """
    with open(output_file, 'w', encoding='utf-8') as fout:
        for sequence in tokenized_sequences:
            fout.write(' '.join(map(str, sequence)) + '\n')


