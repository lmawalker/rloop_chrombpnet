from tensorflow import keras
from chrombpnet.training.utils import augment
from chrombpnet.training.utils import data_utils
import tensorflow as tf
import numpy as np
import random
import string
import math
import os
import json

class ChromBPNetBatchGenerator(keras.utils.Sequence):
    """
    This generator randomly crops (=jitter) and revcomps training examples for
    every epoch, and calls bias model on it, whose outputs (bias profile logits
    and bias logcounts) are fed as input to the chrombpnet model.
    """
    def __init__(self, peak_regions, genome_fasta, batch_size, inputlen, outputlen, max_jitter, cts_bw_file, add_revcomp, return_coords, shuffle_at_epoch_start):
        """
        seqs: B x L' x 4
        cts: B x M'
        inputlen: int (L <= L'), L' is greater to allow for cropping (= jittering)
        outputlen: int (M <= M'), M' is greater to allow for cropping (= jittering)
        batch_size: int (B)
        """

        peak_seqs, peak_cts, peak_coords = data_utils.load_data(peak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter)
        self.peak_seqs = peak_seqs
        self.peak_cts = peak_cts
        self.peak_coords = peak_coords

        self.inputlen = inputlen
        self.outputlen = outputlen
        self.batch_size = batch_size
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.shuffle_at_epoch_start = shuffle_at_epoch_start

        # random crop training data to the desired sizes, revcomp augmentation
        self.crop_revcomp_data()

    def __len__(self):

        return math.ceil(self.seqs.shape[0]/self.batch_size)


    def crop_revcomp_data(self):
        # random crop training data to inputlen and outputlen (with corresponding offsets), revcomp augmentation
        #Sample a fraction of the negative samples according to the specified ratio
        if self.peak_seqs is not None:
            # crop peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords)

            self.seqs = cropped_peaks
            self.cts = cropped_cnts
            self.coords = cropped_coords

        else :
            print("Peak array is empty")

        self.cur_seqs, self.cur_cts, self.cur_coords = augment.crop_revcomp_augment(
                                            self.seqs, self.cts, self.coords, self.inputlen, self.outputlen,
                                            self.add_revcomp, shuffle=self.shuffle_at_epoch_start
                                          )

    def __getitem__(self, idx):
        batch_seq = self.cur_seqs[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_cts = self.cur_cts[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_coords = self.cur_coords[idx*self.batch_size:(idx+1)*self.batch_size]

        if self.return_coords:
            return (batch_seq, [batch_cts, np.log(1+batch_cts.sum(-1, keepdims=True))], batch_coords)
        else:
            return (batch_seq, [batch_cts, np.log(1+batch_cts.sum(-1, keepdims=True))])

    def on_epoch_end(self):
        self.crop_revcomp_data()

