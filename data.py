# Yikang Liao <yikang.liao@tusimple.ai>
# Data module for UCF101

from __future__ import absolute_import
from __future__ import division

import os
import csv
import numpy as np
import mxnet as mx
import random
from videos_reader import sample_clips
import logging


logger = logging.getLogger(__name__)

class ClipBatchIter(mx.io.DataIter):
    def __init__(self, datadir, batch_size=8, n_frame=32, crop_size=112, scale_w=171, scale_h=128, train=True,
                 temporal_center=False):
        super(ClipBatchIter, self).__init__(batch_size)
        self.datadir = datadir
        self.batch_size = batch_size
        self.n_frame = n_frame
        self.crop_size = crop_size
        self.temporal_center = temporal_center
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.train = train
        self.clip_p = 0
        self.clip_lst = []
        self.load_data()
        self.reset()

    @property
    def provide_data(self):
        return [mx.io.DataDesc(name="data", shape=(self.batch_size, 3, self.n_frame, self.crop_size, self.crop_size),
                               dtype=np.float32, layout='NCDHW')]

    @property
    def provide_label(self):
        return [mx.io.DataDesc(name="softmax_label", shape=(self.batch_size,), dtype=np.float32, layout='N')]

    def load_data(self):
        id2class_name = {}
        class_names = []
        with open(os.path.join(self.datadir, 'classInd.txt')) as fin:
            for i, nm in csv.reader(fin, delimiter=' '):
                id2class_name[int(i) - 1] = nm
            for i in xrange(len(id2class_name)):
                class_names.append(id2class_name[i])

        if self.train:
            with open(os.path.join(self.datadir, 'trainlist01.txt')) as fin:
                for nm, c in csv.reader(fin, delimiter=' '):
                    self.clip_lst.append((os.path.join(self.datadir, nm), int(c) - 1))
        else:
            with open(os.path.join(self.datadir, 'testlist01.txt')) as fin:
                for nm, in csv.reader(fin, delimiter=' '):
                    c = nm[:nm.find('/')]
                    self.clip_lst.append((os.path.join(self.datadir, nm), class_names.index(c)))

        logger.info("load data from %s, num clip_lst %d" % (self.datadir, len(self.clip_lst)))


    def reset(self):
        self.clip_p = 0
        if self.train:
            random.shuffle(self.clip_lst)

    def next(self):
        """Get next data batch from iterator.

        Returns
        -------
        DataBatch
            The data of next batch.

        Raises
        ------
        StopIteration
            If the end of the data is reached.
        """

        if self.clip_p < len(self.clip_lst):
            batch_clips = self.clip_lst[self.clip_p: min(len(self.clip_lst), self.clip_p + self.batch_size)]
            # at end of epoch, number of sample remains may be smaller than batch size
            if len(batch_clips) < self.batch_size:
                batch_clips += random.sample(self.clip_lst, self.batch_size-len(batch_clips))
            assert len(batch_clips) == self.batch_size

            filenames, labels = zip(*batch_clips)
            data = sample_clips(filenames, self.batch_size, self.n_frame, self.crop_size, self.scale_w, self.scale_h,
                                self.train, self.temporal_center)
            ret = mx.io.DataBatch([mx.nd.array(data), ], [mx.nd.array(labels), ])
            self.clip_p += self.batch_size
            return ret
        else:
            raise StopIteration






