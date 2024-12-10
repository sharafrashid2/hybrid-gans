import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior
tf.disable_eager_execution()


class ROLLOUT(object):
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate

        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)

        # If using RelGAN, handle differently from SeqGAN (LSTM)
        if getattr(self.lstm, 'use_relgan', False):
            # RelGAN setup
            # Use the relational memory initial state
            self.h0 = self.lstm.init_states  # RMC initial state
            self.g_recurrent_unit = self.create_recurrent_unit_relgan()
            self.g_output_unit = self.create_output_unit_relgan()
        else:
            # SeqGAN (LSTM) setup
            self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0 = tf.stack([self.h0, self.h0])
            self.g_recurrent_unit = self.create_recurrent_unit_seqgan()
            self.g_output_unit = self.create_output_unit_seqgan()

        # Placeholders
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)

        # Processed inputs
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(
                tf.nn.embedding_lookup(self.g_embeddings, self.x),
                perm=[1, 0, 2]
            )  # seq_length x batch_size x emb_dim

        ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.sequence_length, clear_after_read=False)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length, clear_after_read=False)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))

        # gen_x to store generated tokens
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)

        # Depending on RelGAN or SeqGAN, the recurrent unit signature differs:
        # SeqGAN: g_recurrent_unit(x_t, h_tm1) -> h_t (stacked tensor)
        # RelGAN: g_recurrent_unit(x_t, mem_state) -> (mem_o_t, mem_state)
        # We handle both generically by always passing h_tm1 or mem_state.

        if getattr(self.lstm, 'use_relgan', False):
            # For RelGAN, h_tm1 = mem_state initially = self.h0
            # Start token embedding
            start_emb = tf.nn.embedding_lookup(self.g_embeddings, self.start_token)

            # First, run one step to get (mem_o_t, mem_state)
            mem_o_t, mem_state = self.g_recurrent_unit(start_emb, self.h0)

            # When i < given_num, use given tokens
            def _g_recurrence_1(i, mem_o_t, mem_state, given_num, gen_x):
                x_tp1 = ta_emb_x.read(i)  # teacher forced token embedding
                mem_o_t, mem_state = self.g_recurrent_unit(x_tp1, mem_state)
                gen_x = gen_x.write(i, ta_x.read(i))
                return i + 1, mem_o_t, mem_state, given_num, gen_x

            # After given_num, sample tokens
            def _g_recurrence_2(i, mem_o_t, mem_state, given_num, gen_x):
                # Sample from output distribution
                logits = self.g_output_unit((mem_o_t, mem_state))
                log_prob = tf.log(tf.nn.softmax(logits))
                next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
                x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
                gen_x = gen_x.write(i, next_token)
                # Update mem_state for next step
                mem_o_t, mem_state = self.g_recurrent_unit(x_tp1, mem_state)
                return i + 1, mem_o_t, mem_state, given_num, gen_x

            i, mem_o_t, mem_state, given_num, gen_x = tf.while_loop(
                cond=lambda i, _1, _2, given_num, _4: i < given_num,
                body=_g_recurrence_1,
                loop_vars=(tf.constant(0, dtype=tf.int32), mem_o_t, mem_state, self.given_num, gen_x)
            )

            _, _, _, _, gen_x = tf.while_loop(
                cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
                body=_g_recurrence_2,
                loop_vars=(i, mem_o_t, mem_state, given_num, gen_x)
            )

        else:
            # SeqGAN case
            # When i < given_num, use the given tokens
            def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
                h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
                x_tp1 = ta_emb_x.read(i)
                gen_x = gen_x.write(i, ta_x.read(i))
                return i + 1, x_tp1, h_t, given_num, gen_x

            # When i >= given_num, sample tokens
            def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
                h_t = self.g_recurrent_unit(x_t, h_tm1)
                o_t = self.g_output_unit(h_t)  # logits
                log_prob = tf.log(tf.nn.softmax(o_t))
                next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
                x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
                gen_x = gen_x.write(i, next_token)
                return i + 1, x_tp1, h_t, given_num, gen_x

            i, x_t, h_tm1, given_num, gen_x = tf.while_loop(
                cond=lambda i, _1, _2, given_num, _4: i < given_num,
                body=_g_recurrence_1,
                loop_vars=(tf.constant(0, dtype=tf.int32),
                           tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                           self.h0, self.given_num, gen_x)
            )

            _, _, _, _, gen_x = tf.while_loop(
                cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
                body=_g_recurrence_2,
                loop_vars=(i, x_t, h_tm1, given_num, gen_x)
            )

        self.gen_x = gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

    def get_reward(self, sess, input_x, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, self.sequence_length):
                feed = {self.x: input_x, self.given_num: given_num}
                samples = sess.run(self.gen_x, feed)
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # The last token reward
            feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.sequence_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def create_recurrent_unit_seqgan(self):
        # For SeqGAN (LSTM)
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(previous_hidden_state, self.Ui) + self.bi)
            f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(previous_hidden_state, self.Uf) + self.bf)
            o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(previous_hidden_state, self.Uog) + self.bog)
            c_ = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(previous_hidden_state, self.Uc) + self.bc)
            c = f * c_prev + i * c_
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_recurrent_unit_relgan(self):
        # For RelGAN, we assume that the generator provides `gen_mem`
        # Here, we define a unit that takes (x_t, mem_state) and returns (mem_o_t, mem_state)
        def unit(x_t, mem_state):
            mem_o_t, mem_state = self.lstm.gen_mem(x_t, mem_state)
            return mem_o_t, mem_state
        return unit

    def create_output_unit_seqgan(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits
        return unit

    def create_output_unit_relgan(self):
        # For RelGAN, the output unit is already defined in the generator as self.g_output_unit
        # and it takes mem_o_t to produce logits. Here we get mem_o_t from the (mem_o_t, mem_state) tuple.
        def unit(mem_tuple):
            mem_o_t, mem_state = mem_tuple
            logits = self.lstm.g_output_unit(mem_o_t)
            return logits
        return unit

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        if not getattr(self.lstm, 'use_relgan', False):
            # SeqGAN update
            self.g_recurrent_unit = self.update_recurrent_unit_seqgan()
            self.g_output_unit = self.update_output_unit_seqgan()
        else:
            # For RelGAN, if needed, we could define an update mechanism, but typically parameters
            # are updated by the generator itself. Here we just re-set them.
            self.g_recurrent_unit = self.create_recurrent_unit_relgan()
            self.g_output_unit = self.create_output_unit_relgan()

    def update_recurrent_unit_seqgan(self):
        # Updating the weights for SeqGAN (LSTM)
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(previous_hidden_state, self.Ui) + self.bi)
            f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(previous_hidden_state, self.Uf) + self.bf)
            o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(previous_hidden_state, self.Uog) + self.bog)
            c_ = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(previous_hidden_state, self.Uc) + self.bc)
            c = f * c_prev + i * c_
            current_hidden_state = o * tf.nn.tanh(c)
            return tf.stack([current_hidden_state, c])
        return unit

    def update_output_unit_seqgan(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits
        return unit
