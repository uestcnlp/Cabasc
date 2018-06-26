# coding=utf-8
import tensorflow as tf
from MLPAtt_hop_tri import mlpatt_hop
from model.basic_layer.GeneralAttentionLayer import GenAttentionLayer
from model.basic_layer.ConcatAttentionLayer import ConcatAttentionLayer
from model.basic_layer.FwNnAttLayer import FwNnAttLayer
from model.basic_layer.FwNn3AttLayer import FwNnAttLayer as FwNn3AttLayer
from model.basic_layer.DotAttentionLayer import DotAttentionLayer
from model.basic_layer.GatedLayer import GatedLayer


class mlpatt(object):
    def __init__(self, hidden_size, w_shapes, stddev, use_gate=False, is_share_att=True, att_func='fwnn',
                 active='tanh',norm_type="softmax",fwnn_shapes=None,fwnnmlp_shapes = None):
        self.active = active
        self.stddev =stddev
        self.norm_type = norm_type
        self.hidden_size =hidden_size
        if is_share_att:
            if att_func == "dot":
                self.att_layer = DotAttentionLayer(
                    edim=hidden_size,
                )
            elif att_func == "fwnn":
                self.att_layer = FwNnAttLayer(
                    edim=hidden_size,
                    active=active,
                    stddev=stddev,
                    norm_type= norm_type
                )
            elif att_func == "general":
                self.att_layer = GenAttentionLayer(
                    edim=hidden_size,
                    stddev=stddev,
                    norm_type= norm_type
                )
            elif att_func == "concat":
                self.att_layer = ConcatAttentionLayer(
                    edim=hidden_size,
                    stddev=stddev,
                    norm_type= norm_type
                )
            elif att_func == "fwnn3":
                self.att_layer = FwNn3AttLayer(
                    edim=hidden_size,
                    active=active,
                    stddev=stddev,
                    norm_type=norm_type
                )
            else:
                self.att_layer = None

        else:
            self.att_layer = None
        if use_gate:
            if is_share_att:
                self.gated_layer = GatedLayer(hidden_size, stddev)
            else:
                self.gated_layer = None
        else:
            self.gated_layer = None


        if att_func == ["fwnn","fwnn3"]:
            self.att_layer = None
            self.hops = []
            for i in range(0, len(w_shapes)):
                self.hops.append(mlpatt_hop(hidden_size, w_shapes[i], stddev, use_gate,
                                            self.att_layer, self.gated_layer, att_func[i], active, norm_type, fwnn_shapes))
        else:
            self.hops = []
            for i in range(0, len(w_shapes)):
                self.hops.append(mlpatt_hop(hidden_size, w_shapes[i], stddev, use_gate,
                                            self.att_layer, self.gated_layer, att_func, active, norm_type, fwnn_shapes))


    def mlp_compute_tri(self, inputs, aspects, last_output, ctx_bitmap, is_concat_asp=True, is_add=False):
        nhop = len(self.hops)
        alphas = []
        while (nhop):
            hop = self.hops[-nhop]
            nhop = nhop - 1
            last_output, alpha = hop.mlp_forward_tri(inputs, aspects, last_output, ctx_bitmap, is_concat_asp, is_add)
            alphas.append(alpha)
        return last_output, alphas
