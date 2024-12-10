import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense
from relational_memory import RelationalMemory
from ops import *

tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior
tf.disable_eager_execution()

class Generator:
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
             sequence_length, start_token,
             learning_rate=0.01, reward_gamma=0.95, use_relgan=False,
             temperature=1.5, mem_slots=6, head_size=512, num_heads=6):
        """
        Generator with support for both SeqGAN and RelGAN.
        """
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.temperature = temperature
        self.use_relgan = use_relgan
        self.g_params = []  # List to store trainable variables
        self.grad_clip = 5.0  # Gradient clipping threshold
        # Placeholder for rewards (used in training logic)
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])
        if use_relgan:
            self.build_relgan(mem_slots, head_size, num_heads)
        else:
            self.build_seqgan()

        # Add training logic after building the generator
        self.add_training_logic()

    def build_seqgan(self):
        """
        Build SeqGAN generator logic.
        """
        with tf.variable_scope('seqgan_generator', reuse=tf.AUTO_REUSE):
            # Embeddings
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)

            # Recurrent and output units
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
            self.g_output_unit = self.create_output_unit(self.g_params)

            # Placeholders
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])  # Sequence of tokens
            self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])  # Rewards for training

            # Processed inputs for embedding lookup
            with tf.device("/cpu:0"):
                self.processed_x = tf.transpose(
                    tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2]
                )  # Shape: seq_length x batch_size x emb_dim

            # Initial states
            self.h0 = tf.zeros([self.batch_size, self.hidden_dim])  # Initial hidden state
            self.h0 = tf.stack([self.h0, self.h0])  # Hidden state with cell state (for LSTM)

            # TensorArrays for sequence generation
            gen_o = tf.TensorArray(dtype=tf.float32, size=self.sequence_length, infer_shape=True)
            gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length, infer_shape=True)

            # Recurrence function for generating sequences
            def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
                h_t = self.g_recurrent_unit(x_t, h_tm1)  # Compute next hidden state
                o_t = self.g_output_unit(h_t)  # Compute output logits
                log_prob = tf.log(tf.nn.softmax(o_t))  # Convert logits to log probabilities
                next_token = tf.cast(
                    tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32
                )  # Sample next token
                x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # Lookup embedding for next token
                gen_o = gen_o.write(
                    i,
                    tf.reduce_sum(
                        tf.multiply(
                            tf.one_hot(next_token, self.num_emb, 1.0, 0.0), tf.nn.softmax(o_t)
                        ),
                        1,
                    ),
                )  # Store probabilities
                gen_x = gen_x.write(i, next_token)  # Store generated token
                return i + 1, x_tp1, h_t, gen_o, gen_x

            # Generate sequence using while loop
            _, _, _, self.gen_o, self.gen_x = tf.while_loop(
                cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
                body=_g_recurrence,
                loop_vars=(
                    tf.constant(0, dtype=tf.int32),
                    tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                    self.h0,
                    gen_o,
                    gen_x,
                ),
            )

            # Transpose generated sequence to batch_size x seq_length
            self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1, 0])  # Final generated sequence

    def build_relgan(self, mem_slots, head_size, num_heads):
        """
        Build RelGAN generator logic.
        """
        with tf.variable_scope("relgan_generator", reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.Variable(float(0.001), trainable=False)
            # Embedding matrix for input tokens
            self.g_embeddings = tf.get_variable(
                'g_emb',
                shape=[self.num_emb, self.emb_dim],
                initializer=create_linear_initializer(self.num_emb),
                dtype=tf.float32
            )
            self.g_params.append(self.g_embeddings)

            # Relational Memory Core
            self.gen_mem = RelationalMemory(
                mem_slots=mem_slots,
                head_size=head_size,
                num_heads=num_heads,
                name="relational_memory",
            )
            self.g_params.extend(self.gen_mem.rmc_params)  # Include RMC trainable variables

            # Output unit for generating logits
            self.g_output_unit = tf.keras.layers.Dense(self.num_emb, activation=None)

            # Placeholder for input sequences (processed_x)
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])

            # Processed embeddings for input
            with tf.device("/cpu:0"):
                self.processed_x = tf.transpose(
                    tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2]
                )

            # Initial memory state
            self.init_states = self.gen_mem.initial_state(self.batch_size)

            # TensorArrays for generated output and sequences
            gen_o = tf.TensorArray(dtype=tf.float32, size=self.sequence_length, infer_shape=True)
            gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length, infer_shape=True)

            def _gen_recurrence(i, x_t, mem_state, gen_o, gen_x):
                """
                Recurrence step for sequence generation.
                """
                # Pass input and memory through Relational Memory Core
                mem_o_t, mem_state = self.gen_mem(x_t, mem_state)

                # Compute logits for the next token
                o_t = self.g_output_unit(mem_o_t)

                # Apply Gumbel-Softmax for sampling
                gumbel_t = add_gumbel(o_t)
                next_token = tf.stop_gradient(tf.argmax(gumbel_t, axis=1, output_type=tf.int32))

                # Embed the next token
                x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)

                # Write probabilities and tokens to TensorArrays
                gen_o = gen_o.write(
                    i,
                    tf.reduce_sum(
                        tf.multiply(
                            tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                            tf.nn.softmax(tf.multiply(gumbel_t, self.temperature)),
                        ),
                        1,
                    ),
                )
                gen_x = gen_x.write(i, next_token)
                return i + 1, x_tp1, mem_state, gen_o, gen_x

            # Perform sequence generation using tf.while_loop
            _, _, _, self.gen_o, self.gen_x = tf.while_loop(
                cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
                body=_gen_recurrence,
                loop_vars=(
                    tf.constant(0, dtype=tf.int32),
                    tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                    self.init_states,
                    gen_o,
                    gen_x,
                ),
            )

            # Final generated sequences
            self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1, 0])

    def add_training_logic(self):
        """
        Add shared training logic for supervised and unsupervised learning.
        Handles both SeqGAN and RelGAN modes.
        """
        x_emb = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_len x batch_size x emb_dim

        with tf.variable_scope("training_logic", reuse=tf.AUTO_REUSE):
            if self.use_relgan:
                # TensorArrays for storing predictions
                g_predictions = tf.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)

                # TensorArray for processing embedded inputs
                ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
                ta_emb_x = ta_emb_x.unstack(x_emb)

                def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
                    # Forward step using relational memory
                    mem_o_t, h_t = self.gen_mem(x_t, h_tm1)  # Relational memory forward pass
                    o_t = self.g_output_unit(mem_o_t)  # Output logits
                    g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # Write softmax output to TensorArray
                    x_tp1 = ta_emb_x.read(i)  # Move to the next input
                    return i + 1, x_tp1, h_t, g_predictions

                # Run through the sequence with TensorFlow's while_loop
                _, _, _, self.g_predictions = tf.compat.v1.while_loop(
                    cond=lambda i, _1, _2, _3: i < self.sequence_length,
                    body=_pretrain_recurrence,
                    loop_vars=(
                        tf.constant(0, dtype=tf.int32),
                        tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                        self.init_states,  # Relational memory initial state
                        g_predictions
                    )
                )

            else:
                g_predictions = tf.TensorArray(dtype=tf.float32, size=self.sequence_length, infer_shape=True)

                # TensorArray for embedding lookup in sequences
                ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
                ta_emb_x = ta_emb_x.unstack(self.processed_x)

                def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
                    h_t = self.g_recurrent_unit(x_t, h_tm1)  # LSTM hidden state
                    o_t = self.g_output_unit(h_t)  # logits for SeqGAN
                    g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # Write softmax probabilities to predictions
                    x_tp1 = ta_emb_x.read(i)  # Next input token embedding
                    return i + 1, x_tp1, h_t, g_predictions

                # While loop for supervised pretraining
                _, _, _, self.g_predictions = tf.while_loop(
                    cond=lambda i, _1, _2, _3: i < self.sequence_length,
                    body=_pretrain_recurrence,
                    loop_vars=(
                        tf.constant(0, dtype=tf.int32),
                        tf.nn.embedding_lookup(self.g_embeddings, self.start_token),  # Start token embedding
                        self.h0,  # Initial hidden state for SeqGAN
                        g_predictions
                    )
                )

            # Common for both RelGAN and SeqGAN
            self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # [batch_size, seq_length, vocab_size]

            # Pretraining loss
            self.pretrain_loss = -tf.reduce_sum(
                tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.num_emb, 1.0, 0.0) *
                tf.math.log(tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0))
            ) / (self.sequence_length * self.batch_size)

            # Pretraining optimizer
            pretrain_opt = self.g_optimizer(self.learning_rate)
            self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
            self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

            # Unsupervised training loss
            self.g_loss = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.num_emb, 1.0, 0.0) * tf.math.log(
                        tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                    ), axis=1) * tf.reshape(self.rewards, [-1])
            )

            # Unsupervised training updates
            g_opt = self.g_optimizer(self.learning_rate)
            self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
            self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))


    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def pretrain_step(self, sess, x):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.x: x})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)