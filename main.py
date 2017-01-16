
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
import h5py
import time

from model import *

vid_h5 = h5py.File(global_conf['vid-dataset'], 'r')

vid_train_names = vid_h5['VID/train'].keys()
vid_val_names = vid_h5['VID/val'].keys()


def _crop_bilinear(x, bbox, cropsize):
    nsamples = x.shape[0]
    nchan = x.shape[1]
    xrow = x.shape[2]
    xcol = x.shape[3]
    ccol = cropsize[1]
    crow = cropsize[0]

    ccolf = ccol.astype('float32')
    crowf = crow.astype('float32')

    cx = (bbox[:, 1] + bbox[:, 3]) / 2.
    cy = (bbox[:, 0] + bbox[:, 2]) / 2.
    w = bbox[:, 3] - bbox[:, 1]
    h = bbox[:, 2] - bbox[:, 0]
    dx = w / (ccolf - 1.)
    dy = h / (crowf - 1.)

    cx = cx[:, NP.newaxis]
    cy = cy[:, NP.newaxis]
    dx = dx[:, NP.newaxis]
    dy = dy[:, NP.newaxis]

    ca = TT.arange(ccol, dtype='float32')
    cb = TT.arange(crow, dtype='float32')

    mx = cx + dx * (ca.dimshuffle('x', 0) - (ccolf - 1.) / 2.)
    my = cy + dy * (cb.dimshuffle('x', 0) - (crowf - 1.) / 2.)

    a = TT.arange(xcol, dtype=T.config.floatX)
    b = TT.arange(xrow, dtype=T.config.floatX)

    ax = TT.maximum(0, 1 - TT.abs_(a.dimshuffle('x', 0, 'x') - mx.dimshuffle(0, 'x', 1)))
    ax = ax.dimshuffle(0, 'x', 1, 2).repeat(nchan, axis=1).reshape([-1, xcol, ccol])
    by = TT.maximum(0, 1 - TT.abs_(b.dimshuffle('x', 0, 'x') - my.dimshuffle(0, 'x', 1)))
    by = by.dimshuffle(0, 'x', 1, 2).repeat(nchan, axis=1).reshape([-1, xrow, crow])

    bilin = TT.batched_dot(by.transpose(0, 2, 1), TT.batched_dot(x.reshape([-1, xrow, xcol]), ax))

    return bilin.reshape([nsamples, nchan, crow, ccol])


_x = TT.tensor4()
_bbox = TT.matrix()
_cropsize = TT.ivector()
crop_bilinear = T.function(
        [_x, _bbox, _cropsize],
        _crop_bilinear(_x, _bbox, _cropsize),
        allow_input_downcast=True
        )


def fetch_batch(h5, h5_path, keys, batch_size):
    x = NP.zeros((batch_size, 3, 224, 224))
    x_p = NP.zeros((batch_size, 3, 224, 224))
    b = NP.zeros((batch_size, 4))
    b_p = NP.zeros((batch_size, 4))
    b_x = NP.zeros((batch_size, 4))
    c = NP.zeros((batch_size, 1))

    for i in range(batch_size):
        c[i, 0] = RNG.randint(2)

        if c[i, 0] == 1:
            # confidence = 1, p appears in x
            id_x = id_p = RNG.choice(keys)
        else:
            # confidence = 0, p does not appear in x
            id_x, id_p = RNG.choice(keys, 2)

        image_path_x = '%s/%s/image-224-224' % (h5_path, id_x)
        image_path_p = '%s/%s/image-224-224' % (h5_path, id_p)
        anno_path_x = '%s/%s/annotation' % (h5_path, id_x)
        anno_path_p = '%s/%s/annotation' % (h5_path, id_p)
        t1 = RNG.randint(h5[image_path_p].shape[0])
        t2 = RNG.randint(h5[image_path_x].shape[0])

        # p comes from t1 and x comes from t2
        b_p[i] = h5[anno_path_p][t1, :4]
        b[i] = h5[anno_path_x][t2, :4]
        x_p[i] = h5[image_path_p][t1] / 255.
        x[i] = h5[image_path_x][t2] / 255.

    # randomly slightly perturb b_p
    b_p_w = b_p[:, 3] - b_p[:, 1]
    b_p_h = b_p[:, 2] - b_p[:, 0]
    aug_bbox = RNG.uniform(-augment_bbox_scale, augment_bbox_scale, (batch_size, 4))
    b_p[:, [1, 3]] += aug_bbox[:, [1, 3]] * b_p_w[:, NP.newaxis]
    b_p[:, [0, 2]] += aug_bbox[:, [0, 2]] * b_p_h[:, NP.newaxis]
    b_p = NP.clip(b_p, 0, 224)

    # randomly slightly perturb the bounding box for cropping x
    aug_img = RNG.uniform(0, augment_image_bound, (batch_size, 4))
    aug_img[:, [2, 3]] = 224 - aug_img[:, [2, 3]]
    b_x[:, [0, 1]] = NP.clip(aug_img[:, [0, 1]], 0, b[:, [0, 1]])
    b_x[:, [2, 3]] = NP.clip(aug_img[:, [2, 3]], b[:, [2, 3]], 224)

    # crop new x and p
    p = crop_bilinear(x_p, b_p, (56, 56))
    x = crop_bilinear(x, b_x, (224, 224))

    # compute and normalize new b
    b[:, [1, 3]] = (b[:, [1, 3]] - b_x[:, 1:2]) / (b_x[:, 3] - b_x[:, 1])[:, NP.newaxis] * 2 - 1
    b[:, [0, 2]] = (b[:, [0, 2]] - b_x[:, 0:1]) / (b_x[:, 2] - b_x[:, 0])[:, NP.newaxis] * 2 - 1

    return x, p, c, b


patience = 10000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = 2500
validation_batches = 2000

best_params = None
best_validation_loss = NP.inf

done = False
niter = 0

print 'Loading parameters...'
if len(global_conf.get('model-input', "")) > 0:
    model_file = h5py.File(global_conf['model-input'], 'r')
    param_values = []
    for i in range(len(model_file)):
        param_values.append(model_file['%d' % i].value)
    LL.set_all_param_values(out, param_values)

print 'Preparing validation...'

validation_batch_list = []
for i in range(validation_batches):
    print '@%08d' % i
    validation_batch_list.append(fetch_batch(vid_h5, 'VID/val', vid_val_names, 16))

print 'Training...'

while not done:
    ts = time.time()
    tx, tp, tc, tb = fetch_batch(vid_h5, 'VID/train', vid_train_names, batch_size)
    tt = time.time()
    tfetch = tt - ts

    ts = time.time()
    train_c, train_b, train_loss_c, train_loss_b, train_loss = train_func(tx, tp, tc, tb)
    tt = time.time()
    ttrain = tt - ts
    print '#%08d   %11.7f %11.7f %11.7f %6.2fs %6.2fs' % (
            niter, train_loss_c, train_loss_b, train_loss, tfetch, ttrain
            )

    if (niter + 1) % validation_frequency == 0:
        current_validation_loss = 0
        for i, _vb in enumerate(validation_batch_list):
            ts = time.time()
            vx, vp, vc, vb = _vb
            tt = time.time()
            tfetch = tt - ts

            ts = time.time()
            val_c, val_b, val_loss_c, val_loss_b, val_loss = val_func(vx, vp, vc, vb)
            tt = time.time()
            tval = tt - ts

            current_validation_loss += val_loss
            print '@%08d   %11.7f %11.7f %11.7f %6.2fs %6.2fs' % (
                    i, val_loss_c, val_loss_b, val_loss, tfetch, tval
                    )
        current_validation_loss /= validation_batches
        print '#%08d @ %11.7f %11.7f' % (niter, current_validation_loss, best_validation_loss)

        if current_validation_loss < best_validation_loss:
            if current_validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, niter * patience_increase)

            best_params = LL.get_all_param_values(out)
            best_validation_loss = current_validation_loss

            model_file = h5py.File(
                    global_conf['model-intermediate-outputs'] + '.%d' % niter,
                    'w'
                    )
            for i, p in enumerate(best_params):
                model_file['%d' % i] = p
            model_file.close()

        if patience <= niter:
            done = True

    niter += 1


model_file = h5py.File(global_conf['model-output'], 'w')
for i, p in enumerate(best_params):
    model_file['%d' % i] = p
model_file.close()
