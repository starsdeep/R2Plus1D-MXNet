# Yikang Liao <yikang.liao@tusimple.ai>
# Training Module For R2Plus1D Network

import logging
import argparse
import os
import sys
import mxnet as mx
from utils import inspect_net, load_from_caffe2_pkl
from net import create_r3d
from data import ClipBatchIter


def train(args):
    gpus = [int(i) for i in args.gpus.split(',')]
    num_gpus = len(gpus)

    logging.info("number of gpu %d" % num_gpus)

    if len(gpus) == 0:
        kv = None
    else:
        kv = mx.kvstore.create('local')
    logging.info("Running on GPUs: {}".format(gpus))

    # Modify to make it consistent with the distributed trainer
    total_batch_size = args.batch_per_device * num_gpus

    # Round down epoch size to closest multiple of batch size across machines
    epoch_iters = int(args.epoch_size / total_batch_size)
    args.epoch_size = epoch_iters * total_batch_size
    logging.info("Using epoch size: {}".format(args.epoch_size))

    # Create Network
    net = create_r3d(
        num_class=args.num_class,
        no_bias=True,
        model_depth=args.model_depth,
        final_spatial_kernel=7 if args.crop_size == 112 else 14,
        final_temporal_kernel=int(args.n_frame / 8),
        bn_mom=args.bn_mom,
        cudnn_tune=args.cudnn_tune,
        workspace=args.workspace,
    )

    # Load pretrained params
    arg_p = {}
    aux_p = {}
    if args.pretrained:
        arg_p, aux_p = load_from_caffe2_pkl(args.pretrained, net)
        logging.info("load pretrained okay, num of arg_p %d, num of aux_p %d" % (len(arg_p), len(aux_p)))

    # Create Module
    m = mx.module.Module(net, context=[mx.gpu(i) for i in gpus])  # , fixed_param_names=fixed_params)
    if args.plot:
        v = mx.viz.plot_network(net, title='R2Plus1D-train',
                                shape={'data': (total_batch_size, 3, args.n_frame, args.crop_size, args.crop_size)})
        v.render(filename='models/R2Plus1D-train', cleanup=True)

    train_data = mx.io.PrefetchingIter(ClipBatchIter(datadir=args.datadir, batch_size=total_batch_size,
                                                     n_frame=args.n_frame, crop_size=args.crop_size, train=True,
                                                     scale_w=args.scale_w, scale_h=args.scale_h))
    eval_data = mx.io.PrefetchingIter(ClipBatchIter(datadir=args.datadir, batch_size=total_batch_size,
                                                    n_frame=args.n_frame, crop_size=args.crop_size, train=False,
                                                    scale_w=args.scale_w, scale_h=args.scale_h,
                                                    temporal_center=True))

    # Set optimizer
    optimizer = args.optimizer
    optimizer_params = {}
    optimizer_params['learning_rate'] = args.lr
    optimizer_params['momentum'] = args.momentum
    optimizer_params['wd'] = args.wd

    if args.lr_scheduler_step:
        optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(step=args.lr_scheduler_step,
                                                                           factor=args.lr_scheduler_factor)

    m.fit(
        train_data=train_data,
        eval_data=eval_data,
        eval_metric='accuracy',
        epoch_end_callback=mx.callback.do_checkpoint(args.output + '/test', 1),
        batch_end_callback=mx.callback.Speedometer(total_batch_size, 20),
        kvstore=kv,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params=arg_p,
        aux_params=aux_p,
        allow_missing=True,
        begin_epoch=args.begin_epoch,
        num_epoch=args.num_epoch,
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training p3d network")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--plot', type=int, default=0, help='plot the network architecture')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model path')
    parser.add_argument('--datadir', type=str, default='/mnt/truenas/scratch/yijiewang/deep-video/deep-p3d/UCF101/',
                        help='the UCF101 datasets directory')
    parser.add_argument('--output', type=str, default='./output/', help='the output directory')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--cudnn_tune', type=str, default='off', help='optimizer')
    parser.add_argument('--workspace', type=int, default=512, help='workspace for GPU')
    parser.add_argument('--lr_scheduler_step', type=int, default=0, help='reduce lr after n step')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr scheduler factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='initialization learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn_mom', type=float, default=0.9, help='momentum for bn')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
    parser.add_argument('--num_class', type=int, default=101, help='the number of class')
    parser.add_argument('--model_depth', type=int, default=34, help='network depth')
    parser.add_argument('--num_epoch', type=int, default=90, help='the number of epoch')
    parser.add_argument('--epoch_size', type=int, default=100000, help='the number of epoch')
    parser.add_argument('--begin_epoch', type=int, default=0, help='begin training from epoch begin_epoch')
    parser.add_argument('--n_frame', type=int, default=32, help='the number of frame to sample from a video')
    parser.add_argument('--crop_size', type=int, default=112, help='the size of the sampled frame')
    parser.add_argument('--scale_w', type=int, default=171, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=128, help='the rescaled height of image')

    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.output, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)

    train(args)
