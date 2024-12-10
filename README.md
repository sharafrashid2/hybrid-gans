# hybrid-text-gans
6.8611 Final Project

Modified the SeqGAN codebase (LantaoYu/SeqGAN) to support the StepGAN architecture and hybrid architectures called RelSeqGAN and RelStepGAN that
use the relational memory based generator from RelGAN. 

Dependencies:
nltk
numpy
pandas
tensorflow

To train a GAN with just the SeqGAN architecture, in sequence_gan.py, set both is_relgan and is_stepgan variables to False. Set is_relgan to
True to use a relational memory-based generator instead of the LSTM one. Set is_stepgan to True to use a Value Network based
reward update instead of MCTS rollouts. To begin training, simply run sequence_gan.py. Note that the data must be sequences of tokens (numbers)
separated by lines.

To make a vocabulary mapping words to token numbers, run the text_process.py file.

To convert a text file of sequences of words into a sequences of tokens, run tokenize_text_w_vocab.py, making sure to load in the vocab
created in text_process.py.

To calculate average max-BLEU and average self-BLEU scores with a reference set and generated set of sequences, run compare_bleu.py



