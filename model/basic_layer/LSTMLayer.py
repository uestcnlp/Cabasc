import tensorflow as tf
from util.TensorGather import last_relevant


class LSTMLayer(object):
    '''
    the lstm basic_layer.
    '''

    def __init__(
        self,
        hidden_size,
        output_keep_prob=0.8,
        input_keep_prob=1.0,
        forget_bias=1.0,
        cell="lstm"
    ):
        '''
        init the lstm basic_layer.
        hidden_size: the lstm hidden unit size. 
        output_keep_prob: 1 - the output dropout probability. 
        input_keep_prob: 1 - the input dropout probability. 
        forget_bias: the bias of the forget gate. 
        '''
        self.hidden_size = hidden_size
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.forget_bias = forget_bias
        self.first_use = True
        if cell == "gru":
            self.lstm_cell = tf.nn.rnn_cell.GRUCell(
                num_units=self.hidden_size
            )
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.lstm_cell,
                input_keep_prob=self.input_keep_prob,
                output_keep_prob=self.output_keep_prob
            )
        elif cell == "lstm":
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=self.hidden_size,
                forget_bias=self.forget_bias,
                state_is_tuple=True
            )
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.lstm_cell,
                input_keep_prob=self.input_keep_prob,
                output_keep_prob=self.output_keep_prob
            )

    def forward(self, inputs, sequence_length, last_outputs=False,init_state= None,name='1'):
        '''
        inputs: the input of the lstm. 
        inputs.shape = [batch_size, timestep_size, edim]
        sequence_length: the length of each sample. 
        sequence_length = [batch_size]. 
        outputs.shape = [batch_size, timestep_size, hidden_size]
        state.shape = [batch_size, hidden_size]
        last_outputs: is it return the last relevent outputs
        '''
        batch_size = tf.shape(inputs)[0]
        # init the state of lstm
        if init_state is None:
            init_state = self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # run the lstm.
        if self.first_use:
            reuse = None
            self.first_use = False
        else:
            reuse = True
        with tf.variable_scope(name, reuse=reuse):
            output, last_states = tf.nn.dynamic_rnn(
                cell=self.lstm_cell,
                dtype=tf.float32,
                initial_state=init_state,
                sequence_length=sequence_length,
                inputs=inputs,
                time_major=False
            )
            if last_outputs:
                output = last_relevant(output, sequence_length)
                output = tf.reshape(
                    output, [batch_size, 1, self.hidden_size])
        return output, last_states
