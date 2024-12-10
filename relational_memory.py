import tensorflow.compat.v1 as tf
from ops import linear, mlp

# Disable eager execution for TensorFlow 1.x compatibility
tf.disable_eager_execution()

class CustomLayerNormalization(object):
    def __init__(self, dim, epsilon=1e-6, name='layer_norm'):
        self.epsilon = epsilon
        self.name = name
        self.dim = dim
        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable('gamma', shape=[self.dim], initializer=tf.ones_initializer())
            self.beta = tf.get_variable('beta', shape=[self.dim], initializer=tf.zeros_initializer())

    def __call__(self, x):
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

class RelationalMemory(object):
    """Relational Memory Core."""

    def __init__(self, mem_slots, head_size, num_heads=1, num_blocks=1,
                 forget_bias=1.0, input_bias=0.0, gate_style='unit',
                 attention_mlp_layers=2, key_size=None, name='relational_memory'):
        self._mem_slots = mem_slots
        self._head_size = head_size
        self._num_heads = num_heads
        self._mem_size = self._head_size * self._num_heads
        self._name = name

        if num_blocks < 1:
            raise ValueError('num_blocks must be >= 1. Got: {}.'.format(num_blocks))
        self._num_blocks = num_blocks

        self._forget_bias = forget_bias
        self._input_bias = input_bias

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError('gate_style must be one of [\'unit\', \'memory\', None]. Got: {}'.format(gate_style))
        self._gate_style = gate_style

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(attention_mlp_layers))
        self._attention_mlp_layers = attention_mlp_layers

        self._key_size = key_size if key_size else self._head_size

        # Template for variable sharing
        self._template = tf.make_template(self._name, self._build)

        # Initialize two LayerNormalization instances here with appropriate dimensions
        qkv_size = 2 * self._key_size + self._head_size
        total_size = qkv_size * self._num_heads

        self.layer_norm1 = CustomLayerNormalization(dim=total_size, name='layer_norm1')
        self.layer_norm2 = CustomLayerNormalization(dim=self._mem_size, name='layer_norm2')

        # Force variable creation by calling functions with dummy inputs
        dummy_batch_size = 1
        dummy_memory = tf.zeros([dummy_batch_size, self._mem_slots, self._mem_size])
        dummy_inputs = tf.zeros([dummy_batch_size, self._mem_size])
        self._build(dummy_inputs, dummy_memory)

    def initial_state(self, batch_size):
        init_state = tf.eye(self._mem_slots, batch_shape=[batch_size])

        if self._mem_size > self._mem_slots:
            difference = self._mem_size - self._mem_slots
            pad = tf.zeros((batch_size, self._mem_slots, difference))
            init_state = tf.concat([init_state, pad], -1)
        elif self._mem_size < self._mem_slots:
            init_state = init_state[:, :, :self._mem_size]
        return init_state

    def _multihead_attention(self, memory):
        qkv_size = 2 * self._key_size + self._head_size
        total_size = qkv_size * self._num_heads
        batch_size = tf.shape(memory)[0]
        memory_flattened = tf.reshape(memory, [-1, self._mem_size])
        qkv = linear(memory_flattened, total_size, use_bias=False, scope='lin_qkv')
        qkv = tf.reshape(qkv, [batch_size, -1, total_size])
        qkv = self.layer_norm1(qkv)  # Use the first layer normalization

        qkv_reshape = tf.reshape(qkv, [batch_size, -1, self._num_heads, qkv_size])
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(qkv_transpose, [self._key_size, self._key_size, self._head_size], -1)

        q *= qkv_size ** -0.5
        dot_product = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(dot_product)

        output = tf.matmul(weights, v)
        output_transpose = tf.transpose(output, [0, 2, 1, 3])
        new_memory = tf.reshape(output_transpose, [batch_size, -1, self._mem_size])
        return new_memory

    @property
    def state_size(self):
        return tf.TensorShape([self._mem_slots, self._mem_size])

    @property
    def output_size(self):
        return tf.TensorShape(self._mem_slots * self._mem_size)

    def _calculate_gate_size(self):
        if self._gate_style == 'unit':
            return self._mem_size
        elif self._gate_style == 'memory':
            return 1
        else:
            return 0

    def _create_gates(self, inputs, memory):
        num_gates = 2 * self._calculate_gate_size()
        batch_size = tf.shape(memory)[0]

        memory = tf.tanh(memory)

        inputs = tf.reshape(inputs, [batch_size, -1])
        gate_inputs = linear(inputs, num_gates, use_bias=False, scope='gate_in')
        gate_inputs = tf.expand_dims(gate_inputs, axis=1)

        memory_flattened = tf.reshape(memory, [-1, self._mem_size])
        gate_memory = linear(memory_flattened, num_gates, use_bias=False, scope='gate_mem')
        gate_memory = tf.reshape(gate_memory, [batch_size, self._mem_slots, num_gates])

        gates = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=2)
        input_gate, forget_gate = gates

        input_gate = tf.sigmoid(input_gate + self._input_bias)
        forget_gate = tf.sigmoid(forget_gate + self._forget_bias)

        return input_gate, forget_gate

    def _attend_over_memory(self, memory):
        for _ in range(self._num_blocks):
            attended_memory = self._multihead_attention(memory)
            memory = self.layer_norm2(memory + attended_memory)  # Use the second layer normalization

            batch_size = tf.shape(memory)[0]
            memory_mlp = tf.reshape(memory, [-1, self._mem_size])
            memory_mlp = mlp(memory_mlp, [self._mem_size] * self._attention_mlp_layers)
            memory_mlp = tf.reshape(memory_mlp, [batch_size, -1, self._mem_size])

            memory = self.layer_norm2(memory + memory_mlp)  # Use the second layer normalization

        return memory

    def _build(self, inputs, memory):
        batch_size = tf.shape(memory)[0]
        inputs = tf.reshape(inputs, [batch_size, -1])
        inputs = linear(inputs, self._mem_size, use_bias=True, scope='input_for_concat')
        inputs_reshape = tf.expand_dims(inputs, 1)

        memory_plus_input = tf.concat([memory, inputs_reshape], axis=1)
        next_memory = self._attend_over_memory(memory_plus_input)

        n = tf.shape(inputs_reshape)[1]
        next_memory = next_memory[:, :-n, :]

        if self._gate_style == 'unit' or self._gate_style == 'memory':
            self._input_gate, self._forget_gate = self._create_gates(inputs_reshape, memory)
            next_memory = self._input_gate * tf.tanh(next_memory)
            next_memory += self._forget_gate * memory

        output = tf.reshape(next_memory, [batch_size, -1])
        return output, next_memory

    def __call__(self, *args, **kwargs):
        outputs = self._template(*args, **kwargs)
        return outputs

    @property
    def input_gate(self):
        return self._input_gate

    @property
    def forget_gate(self):
        return self._forget_gate

    @property
    def rmc_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

    def set_rmc_params(self, ref_rmc_params):
        rmc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        if len(rmc_params) != len(ref_rmc_params):
            raise ValueError("the number of parameters in the two RMC modules does not match")
        for i in range(len(ref_rmc_params)):
            rmc_params[i] = tf.identity(ref_rmc_params[i])

    def update_rmc_params(self, ref_rmc_params, update_ratio):
        rmc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        if len(rmc_params) != len(ref_rmc_params):
            raise ValueError("the number of parameters in the two RMC modules does not match")
        for i in range(len(ref_rmc_params)):
            rmc_params[i] = update_ratio * rmc_params[i] + (1 - update_ratio) * tf.identity(ref_rmc_params[i])
    
    @property
    def trainable_variables(self):
        """
        Collect all trainable variables within the scope of the RelationalMemory module.
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
