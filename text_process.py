# coding=utf-8
import json
import nltk
nltk.download('punkt')

def preprocess_dataset_from_lines(train_file, output_tokenized_file, vocab_file, seq_len=None):
    """
    Preprocess the dataset where sequences are separated by newlines.

    Args:
        train_file: Path to the input training dataset.
        output_tokenized_file: Path to save the tokenized numerical sequences.
        vocab_file: Path to save the vocabulary dictionary (word-to-index).
        seq_len: Fixed sequence length. If None, calculate from the dataset.

    Returns:
        word_index_dict: Mapping of words to indices.
        index_word_dict: Mapping of indices to words.
        seq_len: The sequence length used for padding.
    """
    # Step 1: Read and tokenize each line (sequence)
    train_tokens = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = nltk.word_tokenize(line.strip().lower())
            train_tokens.append(tokens)

    # Step 2: Create vocabulary and mappings
    word_set = get_word_list(train_tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    # Step 3: Determine sequence length
    if seq_len is None:
        seq_len = len(max(train_tokens, key=len))  # Longest sequence in the dataset

    # Step 4: Convert tokenized text to numerical sequences
    tokenized_code = text_to_code(train_tokens, word_index_dict, seq_len)

    # Step 5: Save tokenized numerical sequences
    with open(output_tokenized_file, 'w', encoding='utf-8') as f:
        f.write(tokenized_code)

    # Step 6: Save the vocabulary dictionary
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(word_index_dict, f)

    print(f"Tokenized data saved to {output_tokenized_file}")
    print(f"Vocabulary saved to {vocab_file}")
    return word_index_dict, index_word_dict, seq_len



# text tokens to code strings
def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)  # used to filled in the blank to make up a sentence with seq_len
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


# code tokens to text strings
def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


# tokenlize the file
def get_tokenlized(file):
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


# get word set
def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


# get word_index_dict and index_word_dict
def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


# get sequence length and dict size
def text_precess(train_text_loc, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    # with open(oracle_file, 'w') as outfile:
    #     outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return sequence_len, len(word_index_dict) + 1

# Combine train and test datasets
def create_vocabulary(train_file, test_file):
    train_tokens = get_tokenlized(train_file)
    test_tokens = get_tokenlized(test_file)

    # Combine tokens from both datasets
    combined_tokens = train_tokens + test_tokens

    # Create the word set
    word_set = get_word_list(combined_tokens)

    # Create dictionaries
    word_to_index, index_to_word = get_dict(word_set)

    # Determine maximum sequence length
    max_seq_len = max(
        len(max(train_tokens, key=len, default=[])),
        len(max(test_tokens, key=len, default=[]))
    )

    return word_to_index, index_to_word, max_seq_len
    

# Example Usage
train_file = "save/prompt_injection.txt"
test_file = "save/test_prompt_injection.txt"

word_to_index, index_to_word, max_seq_len = create_vocabulary(train_file, test_file)

# Save the dictionaries for future use
import pickle
with open("prompt_injection_word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)
with open("prompt_injection_index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)

print(f"Vocabulary size: {len(word_to_index)}")
print(f"Maximum sequence length: {max_seq_len}")