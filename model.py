
import theano as T
import theano.tensor as TT
from theano.compile.nanguardmode import NanGuardMode

import lasagne as L
import lasagne.layers as LL
import lasagne.updates as LU
import lasagne.init as LI

import numpy as NP
import numpy.random as RNG
import json
import sys


def lossfunc_gaussian(b, b_):
    return 0.5 * ((b_ - b) ** 2)

def lossfunc_laplace(b, b_):
    return TT.abs_(b_ - b)

def lossfunc_smooth(b, b_):
    return TT.switch(TT.abs_(b_ - b) > 2, TT.abs_(b_ - b), 0.5 * (b_ - b) ** 2)

actv = {
        "sigm": L.nonlinearities.sigmoid,
        "tanh": L.nonlinearities.tanh,
        "relu": L.nonlinearities.rectify,
        "identity": L.nonlinearities.identity,
        }
lossfunc_b = {
        "gaussian": lossfunc_gaussian,
        "laplace": lossfunc_laplace,
        "smooth": lossfunc_smooth,
        }


net_conf_f = open(sys.argv[1])
net_conf = json.load(net_conf_f)
net_conf_f.close()

branch_conf = {}

for _conf in net_conf:
    if _conf['branch'] == '':
        global_conf = _conf
    else:
        branch_conf[_conf['branch']] = _conf['structure']
batch_size = global_conf['batch-size']
augment_bbox_scale = global_conf['augment-bbox-scale']
augment_image_bound = global_conf['augment-image-bound']
relu_bias_init = global_conf['relu-bias-init']
if global_conf.get('cudnn', False):
    if global_conf.get('libgpuarray', False):
        import dnn_gpuarray as LLDNN
    else:
        import lasagne.layers.dnn as LLDNN
    Conv2DLayer = LLDNN.Conv2DDNNLayer
    Pool2DLayer = LLDNN.Pool2DDNNLayer
    batch_norm = LLDNN.batch_norm_dnn
else:
    Conv2DLayer = LL.Conv2DLayer
    Pool2DLayer = LL.Pool2DLayer
    batch_norm = LL.batch_norm

def parse_branch_conf(start, branch_name):
    cur = start
    params = []
    for _conf in branch_conf[branch_name]:
        if _conf['type'] == 'conv':
            new = Conv2DLayer(
                    cur,
                    num_filters=_conf['channels'],
                    filter_size=_conf['size'],
                    nonlinearity=actv[_conf['actv']],
                    b=LI.Constant(relu_bias_init if _conf['actv'] == 'relu' else 0),
                    pad='same',
                    )
        elif _conf['type'] == 'pool':
            new = Pool2DLayer(
                    cur,
                    mode=_conf['mode'],
                    pool_size=_conf['stride'],
                    )
        elif _conf['type'] == 'dense':
            new = LL.DenseLayer(
                    cur,
                    num_units=_conf['outputs'],
                    nonlinearity=actv[_conf['actv']],
                    b=LI.Constant(relu_bias_init if _conf['actv'] == 'relu' else 0),
                    )
        elif _conf['type'] == 'flatten':
            new = LL.FlattenLayer(cur)

        if _conf.get('bn', False):
            new = batch_norm(
                    new,
                    beta=LI.Constant(relu_bias_init if _conf['actv'] == 'relu' else 0)
                    )
            if isinstance(new, LL.NonlinearityLayer):
                _bnlayer = new.input_layer
            else:
                _bnlayer = new
            _cnnlayer = _bnlayer.input_layer

            params.extend(_bnlayer.get_params(trainable=True))
            params.extend(_cnnlayer.get_params(trainable=True))
        else:
            params.extend(new.get_params(trainable=True))

        cur = new

    return cur, params


x = TT.tensor4()    # (batch_size, 3, 224, 224)
p = TT.tensor4()    # (batch_size, 3, 56, 56)
c = TT.col()        # (batch_size, 1)
b = TT.matrix()     # (batch_size, 4)


x_in = LL.InputLayer(shape=(None, 3, 224, 224), input_var=x)
p_in = LL.InputLayer(shape=(None, 3, 56, 56), input_var=p)
x_out, x_params = parse_branch_conf(x_in, 'x')
p_out, p_params = parse_branch_conf(p_in, 'p')
f_in = LL.ConcatLayer([x_out, p_out])
f_out, f_params = parse_branch_conf(f_in, 'F')

c_out = LL.DenseLayer(f_out, num_units=1, nonlinearity=actv['sigm'])
b_out = LL.DenseLayer(f_out, num_units=4, nonlinearity=actv['identity'])
out = LL.ConcatLayer([c_out, b_out])

c_ = LL.get_output(c_out)
b_ = LL.get_output(b_out)

loss_c = -(c * TT.log(c_) + (1 - c) * TT.log(1 - c_)).mean()
loss_b = TT.switch(
        TT.eq(c.sum(), 0),
        0,
        (c * lossfunc_b[global_conf['loss-b']](b, b_)).sum() / c.sum()
        )
loss = loss_c + loss_b

params = LL.get_all_params(out, trainable=True)
#params = f_params
grads = T.grad(loss, params)
constrained_grads = LU.total_norm_constraint(grads, 1)
updates = LU.adam(constrained_grads, params)

if global_conf.get('debug', False):
    train_func = T.function(
            [x, p, c, b],
            [c_, b_, loss_c, loss_b, loss],
            updates=updates,
            allow_input_downcast=True,
            mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
            )
else:
    train_func = T.function(
            [x, p, c, b],
            [c_, b_, loss_c, loss_b, loss],
            updates=updates,
            allow_input_downcast=True,
            )

val_func = T.function(
        [x, p, c, b],
        [c_, b_, loss_c, loss_b, loss],
        allow_input_downcast=True,
        )
test_func = T.function(
        [x, p],
        [c_, b_],
        allow_input_downcast=True,
        )
