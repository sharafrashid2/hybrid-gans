import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import LSTMCell, RNN, Dense
tf.disable_v2_behavior()
import numpy as np

class ValueNetwork:
    def __init__(self, sequence_length, hidden_dim, vocab_size):
        """
        Initialize the Value Network.
        Args:
            sequence_length: The length of the generated sequences.
            hidden_dim: Hidden dimension for the LSTM.
            vocab_size: Vocabulary size.
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Placeholders
        self.input_sequences = tf.placeholder(tf.int32, [None, sequence_length], name="input_sequences")
        self.target_values = tf.placeholder(tf.float32, [None, sequence_length], name="target_values")

        # Value Network Embeddings
        with tf.variable_scope("value_network"):
            self.embeddings = tf.Variable(
                tf.random_uniform([vocab_size, hidden_dim], -1.0, 1.0),
                name="value_embeddings"
            )
            self.processed_input = tf.nn.embedding_lookup(self.embeddings, self.input_sequences)

            # LSTM for processing sequences
            lstm_cell = LSTMCell(hidden_dim)
            rnn_layer = RNN(lstm_cell, return_sequences=True, return_state=True)
            self.outputs, *self.states = rnn_layer(self.processed_input)

            # Dense layer to compute predicted values
            self.logits = Dense(1, activation='tanh')(self.outputs)  # Predict values in a bounded range
            self.predicted_values = tf.squeeze(self.logits, axis=-1)  # Shape: (batch_size, sequence_length)

            # Loss and optimization
            self.loss = tf.reduce_mean(tf.square(self.predicted_values - self.target_values))  # MSE Loss
            self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def compute_q_values(self, sess, sequences, discriminator):
        """
        Compute Q-values for a batch of sequences.
        Args:
            sess: TensorFlow session.
            sequences: Input sequences.
            discriminator: Discriminator model for computing rewards.
        Returns:
            Stepwise Q-values for the input sequences.
        """
        feed = {discriminator.input_x: sequences, discriminator.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)

        # Generate stepwise rewards (e.g., penalize repetition or encourage diversity)
        q_values = np.zeros((len(sequences), self.sequence_length))
        for i, sequence in enumerate(sequences):
            reward = ypred_for_auc[i][1]  # Positive class probability
            for t in range(self.sequence_length):
                # Penalize repeating tokens
                if isinstance(sequence, list):
                  # For Python lists (e.g., SeqGAN)
                  token_count = sequence[:t+1].count(sequence[t])
                elif isinstance(sequence, np.ndarray):
                    # For NumPy arrays (e.g., RelGAN)
                    token_count = np.count_nonzero(sequence[:t+1] == sequence[t])
                else:
                    raise TypeError("Sequence type not supported. Expected list or numpy.ndarray.")
                    q_values[i, t] = reward / (1 + token_count)  # Penalize repetitions
        return q_values

    def train(self, sess, sequences, target_values):
        """
        Train the Value Network on the given sequences and target values.
        Args:
            sess: TensorFlow session.
            sequences: Input sequences.
            target_values: Target values (e.g., Q-values from discriminator).
        Returns:
            Training loss.
        """
        # Ensure target_values has the correct shape
        target_values = np.asarray(target_values)
        
        # If target_values has more columns than sequence_length, slice it
        if target_values.shape[1] > self.sequence_length:
            target_values = target_values[:, :self.sequence_length]
        # If target_values has fewer columns, tile or pad it
        elif target_values.shape[1] < self.sequence_length:
            target_values = np.tile(target_values, (1, self.sequence_length // target_values.shape[1] + 1))[:, :self.sequence_length]

        # Feed data to the model
        feed = {self.input_sequences: sequences, self.target_values: target_values}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed)
        return loss 

    def predict(self, sess, sequences):
        """
        Predict values for a batch of sequences.
        Args:
            sess: TensorFlow session.
            sequences: Input sequences.
        Returns:
            Predicted values for the sequences.
        """
        feed = {self.input_sequences: sequences}
        predicted_values = sess.run(self.predicted_values, feed_dict=feed)
        return predicted_values
