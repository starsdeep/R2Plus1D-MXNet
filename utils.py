# Yikang Liao <yikang.liao@tusimple.ai>
# Utility Functions

import cv2
import logging
import cPickle as pickle
from pprint import pprint
import mxnet as mx
logger = logging.getLogger(__name__)


def load_from_caffe2_pkl(filepath, net):
    args_loaded = {}
    auxs_loaded = {}

    with open(filepath, 'r') as fopen:
        blobs = pickle.load(fopen)['blobs']
    print("len of blobs %d" % len(blobs))

    for k, v in blobs.iteritems():
        if k.endswith('_w'):
            args_loaded[k[:-2] + '_weight'] = mx.nd.array(v)
        if k.endswith('_b'):
            args_loaded[k[:-2] + '_beta'] = mx.nd.array(v)
        if k.endswith('_s'):
            args_loaded[k[:-2] + '_gamma'] = mx.nd.array(v)

        if k.endswith('_rm'):
            auxs_loaded[k[:-3] + '_moving_mean'] = mx.nd.array(v)
        if k.endswith('_riv'):
            auxs_loaded[k[:-4] + '_moving_var'] = mx.nd.array(1.0 / v)


    args_symbol = net.list_arguments()
    auxs_symbol = net.list_auxiliary_states()
    logger.info("symbol has %d = %d arg + %d aux" % (len(args_symbol)+len(auxs_symbol), len(args_symbol), len(auxs_symbol)))
    logger.info("model loaded has %d = %d arg + %d aux" % (len(args_loaded)+len(auxs_loaded), len(args_loaded), len(auxs_loaded)))

    logger.info("testing arg loaded")
    for arg in args_symbol:
        if arg not in args_loaded:
            logger.info("arg %s not loaded" % arg)

    logger.info("testing arg used in net")
    for arg in args_loaded:
        if arg not in args_symbol:
            logger.info("arg %s not used in net" % arg)

    logger.info("testing aux")
    for aux in auxs_symbol:
        if aux not in auxs_loaded:
            logger.info("aux %s not loaded" % aux)

    return args_loaded, auxs_loaded


def inspect_net(net):
    pprint("name %s" % net.name)
    print("===========%d of arg============" % len(net.list_arguments()))
    pprint(net.list_arguments())
    print("===========%d of aux============" % len(net.list_auxiliary_states()))
    pprint(net.list_auxiliary_states())
    pprint(net.list_attr())



def test_clip(n_frame=16, size=112):
    v = cv2.VideoCapture('/mnt/truenas/scratch/yijiewang/deep-video/deep-p3d/UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c02.avi')
    width, height, length = v.get(cv2.CAP_PROP_FRAME_WIDTH), v.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                                v.get(cv2.CAP_PROP_FRAME_COUNT)
    print(width, height, length)

    assert n_frame <= length and size <= width and size <= height, \
        '%d <= %d ; %d <= %d ; %d <= %d' % (n_frame, length, size, width, size, height)

    frame_st = random.randrange(length - n_frame + 1)
    row_st = random.randrange(scale_ - size + 1)
    col_st = random.randrange(width - size + 1)
    tmp = np.zeros((n_frame, size, size, 3), dtype=np.float32)
    v.set(cv2.CAP_PROP_POS_FRAMES, frame_st)
    n=0
    t = None
    for frame_p in xrange(n_frame):
        _, f = v.read()
        if f is not None:
            f1 = cv2.resize(f, (171,128))
            t = f1[row_st:row_st+size, col_st:col_st+size, :]
            tmp[frame_p, ...] = f1[row_st:row_st+size, col_st:col_st+size, :]
        else:
            tmp = None
            break

    cv2.imshow('frame0', tmp[0].astype('uint8'))
    cv2.waitKey(0)
    tmp1 = np.flip(tmp, 2)
    cv2.imshow('frame1', tmp1[0].astype('uint8'))
    cv2.waitKey(0)
    # if random.choice([True, False]):
    tmp = tmp / 255.0 * 2.0 - 1.0
    v.release()
    cv2.destroyAllWindows()

