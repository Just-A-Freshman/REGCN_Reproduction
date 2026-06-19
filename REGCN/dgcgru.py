import tensorflow as tf

try:
    from tensorflow.keras.layers import AbstractRNNCell as RNNC
except ImportError:
    from tensorflow.keras.layers import Layer as RNNC
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.constraints import MinMaxNorm

class gcgru(RNNC):

    # def __init__(self, num_units, adj1,adj2, num_gcn_nodes, s_index, **kwargs):
    def __init__(self, num_units, adj, num_gcn_nodes, s_index,
                 kernel_regularizer=None, **kwargs):
        super(gcgru, self).__init__(**kwargs)
        self.units = num_units
        self._gcn_nodes = num_gcn_nodes
        self.s_index = s_index
        self._kernel_regularizer = kernel_regularizer

        self._adj = adj

    @ property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        if input_shape[-1] != self._gcn_nodes:
            raise ValueError(
                f"gcgru: input feature dim ({input_shape[-1]}) does not match "
                f"n_gcn_nodes ({self._gcn_nodes}). Set n_gcn_nodes to match "
                f"the number of feature columns in your data."
            )
        reg = self._kernel_regularizer
        # weights
        self.wz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='wz')
        self.wr = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='wr')
        self.wh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='wh')

        self.w0 = self.add_weight(shape=(1, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='w0')
        self.wa = self.add_weight(shape=(self._adj.shape[0],self._gcn_nodes,self._gcn_nodes),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                                  # constraint= UnitNorm(axis=0),
                                  name='wa')
        # us
        self.uz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='uz')
        self.ur = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='ur')
        self.uh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  regularizer=reg,
                                  trainable=True,
                                  name='uh')

        # biases (no regularizer by convention)
        self.bz = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bz")
        self.br = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="br")
        self.bh = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bh")
        self.built = True

    def call(self, inputs, states):
        state = states[0]
        adj = self._adj
        integrated_inputs = tf.math.multiply(self.wa, adj)
        adj_max = tf.reduce_sum(integrated_inputs, axis=0)

        #GCN
        x = self.gc(inputs, adj_max)
        #GRU
        z = K.dot(x, self.wz) + K.dot(state, self.uz) + self.bz
        z = sigmoid(z)
        r = K.dot(x, self.wr) + K.dot(state, self.ur) + self.br
        r = sigmoid(r)
        h = K.dot(x, self.wh) + K.dot((r * state), self.uh) + self.bh
        h = tanh(h)

        output = z * state + (1 - z) * h
        return output, output

    def gc(self, inputs, adj):
        ax = K.dot(inputs, adj)
        ax = ax[:, self.s_index]
        ax = tf.expand_dims(ax, -1)
        return K.dot(ax, self.w0)


