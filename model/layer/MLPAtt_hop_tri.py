import tensorflow as tf
from model.basic_layer.LinearLayer import LinearLayer
from model.basic_layer.FwNnAttLayer import FwNnAttLayer
from model.basic_layer.FwNn3AttLayer import FwNnAttLayer as FwNn3AttLayer
from model.basic_layer.DotAttentionLayer import DotAttentionLayer
from model.basic_layer.LSTMLayer import LSTMLayer
from util.Activer import activer
from model.basic_layer.GatedLayer import GatedLayer
class mlpatt_hop(object):
    def __init__(self, hidden_size, w_shape, stddev,  use_gate = False,
                 att_layer = None, gate_layer = None, att = 'fwnn',active = 'tanh',norm_type='softmax',fwnn_shapes=None):
        self.active = active
        self.stddev= stddev
        self.hidden_size =hidden_size
        if att_layer != None:
            self.att_layer = att_layer
        else:
            if att == "dot":
                self.att_layer = DotAttentionLayer(
                    edim=hidden_size,
                )
            elif att == "fwnn":
                self.att_layer = FwNnAttLayer(
                    edim=hidden_size,
                    active=self.active,
                    stddev=stddev,
                    norm_type=norm_type
                )
            elif att == "fwnn3":
                self.att_layer = FwNn3AttLayer(
                    edim=hidden_size,
                    active=self.active,
                    stddev=stddev,
                    norm_type = norm_type
                )
        if use_gate:
            if gate_layer != None:
                self.gate_layer = gate_layer
            else:
                self.gate_layer = GatedLayer(hidden_size,stddev)
        if w_shape == None:
            self.linear = None
        else:
            self.linear= LinearLayer(w_shape, stddev)


    def mlp_forward_tri(self, inputs, aspects,last_output, ctx_bitmap,is_concat_asp = True, is_add = False):
        batch_size = tf.shape(inputs)[0]
        vec, alpha = self.att_layer.forward(
                context = inputs,
                aspect= aspects,
                output = last_output,
                ctx_bitmap = ctx_bitmap
            )
        vec = tf.reshape(vec, [batch_size, -1])
        if is_add:
            vec = vec + last_output
        if is_concat_asp:
            last_output = tf.concat(axis=1, values=[vec, aspects])
        else:
            last_output = vec
        linear = self.linear
        # nlayer = nlayer -1
        res = linear.forward(last_output)
        last_output = activer(res, self.active)
        return last_output, alpha
