import argparse
import pyfaidx
import pyBigWig
import pandas as pd
import numpy as np
import os
import json
import chrombpnet.helpers.hyperparameters.param_utils as param_utils

def parse_data_args():
    parser=argparse.ArgumentParser(description="find hyper-parameters for chrombpnet defined in src/training/models/chrombpnet_with_bias_model.py")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-i", "--bigwig", type=str, required=True, help="Bigwig")
    parser.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-b", "--bias-threshold-factor", type=float, default=0.5, help="A threshold is applied on maximum count of non-peak region for training bias model, which is set as this threshold x min(count over peak regions)")
    parser.add_argument("-oth", "--outlier-threshold", type=float, default=0.9999, help="threshold to use to filter outlies")
    parser.add_argument("-j", "--max-jitter", type=int, default=50, help="Maximum jitter applied on either side of region (default 500 for chrombpnet model)")
    parser.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    return parser

def parse_model_args(parser):
    # arguments here defined the following model - src/training/models/chrombpnet_with_bias_model.py
    parser.add_argument("-il", "--inputlen", type=int, help="Sequence input length")
    parser.add_argument("-ol", "--outputlen", type=int, help="Prediction output length")
    parser.add_argument("-fil", "--filters", type=int, default=128, help="Number of filters to use in chrombpnet mode")
    parser.add_argument("-dil", "--n-dilation-layers", type=int, default=4, help="Number of dilation layers to use in chrombpnet model")
    parser.add_argument("-op", "--output-prefix", required=True, help="output prefix for storing hyper-param TSV for chrombpnet")
    args = parser.parse_args()
    return args

def main(args):
    # read the fold information - we will evaluate hyperparams and filter outliers on the train+valid set
    # do nothing on the test set
    splits_dict = json.load(open(args.chr_fold_path))
    chroms_to_keep = splits_dict["train"] + splits_dict["valid"]
    test_chroms_to_keep = splits_dict["test"]
    print("evaluating hyperparameters on the following chromosomes", chroms_to_keep)

    # read from bigwigs and fasta file
    bw = pyBigWig.open(args.bigwig)
    genome = pyfaidx.Fasta(args.genome)

    # read peaks and non peaks
    in_peaks = pd.read_csv(args.peaks,
                           sep='\t',
                           header=None,
                           names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])

    assert (in_peaks.shape[0] != 0)  # peaks file is empty
    assert (args.inputlen >= args.outputlen)  # inputlen should be greater than the outputlen
    # inputlen and outlen are chosen based on the filters and dilations layers used

    # get train/valid peaks and test peaks seperately
    peaks = in_peaks[(in_peaks["chr"].isin(chroms_to_keep))]
    test_peaks = in_peaks[(in_peaks["chr"].isin(test_chroms_to_keep))]

    # step 1 filtering: filter peaks that are in the edges - prevents us from making the inputlen regions - do this for all train/test/valid
    peaks = param_utils.filter_edge_regions(peaks, bw, args.inputlen, peaks_bool=1)
    test_peaks = param_utils.filter_edge_regions(test_peaks, bw, args.inputlen, peaks_bool=1)

    # step 2 filtering: filter peaks that have counts less than a threshold_factor (minimum of peak counts)
    peak_cnts, _ = param_utils.get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
    assert (len(peak_cnts) == peaks.shape[0])

    '''import matplotlib.pyplot as plt
    plt.hist(peak_cnts,50)
    plt.xlabel("Peak Counts")
    plt.ylabel("# of Peaks")
    plt.title("Distribution of Peak Counts")
    plt.show()
    '''
    final_cnts = peak_cnts
    counts_threshold = np.max(final_cnts)
    assert (counts_threshold > 0)  # counts threshold is 0 - all non peaks will be filtered!

    upper_thresh = np.max(final_cnts)
    lower_thresh = 0.0

    # combine train valid and test peak set and store them in a new file
    frames = [peaks, test_peaks]
    all_peaks = pd.concat(frames)
    all_peaks.to_csv("{}filtered.bias_peaks.bed".format(args.output_prefix), sep="\t", header=False, index=False)

    # find counts loss weight for model training - using train and validation set
    counts_loss_weight = np.median(final_cnts) / 10
    print("counts_loss_weight:", counts_loss_weight)
    assert (counts_loss_weight != 0)

    if counts_loss_weight < 1.0:
        counts_loss_weight = 1.0
        print("WARNING: you are training on low-read depth data")

        # store the parameters being used  - in a TSV file
        file = open("{}bias_data_params.tsv".format(args.output_prefix), "w")
        file.write("\t".join(["counts_sum_min_thresh", str(round(lower_thresh, 2))]))
        file.write("\n")
        file.write("\t".join(["counts_sum_max_thresh", str(round(upper_thresh, 2))]))
        file.write("\n")
        file.write("\t".join(
            ["trainings_pts_post_thresh", str(sum((final_cnts < upper_thresh) & (final_cnts > lower_thresh)))]))
        file.write("\n")
        file.close()

        file = open("{}bias_model_params.tsv".format(args.output_prefix), "w")
        file.write("\t".join(["counts_loss_weight", str(round(counts_loss_weight, 2))]))
        file.write("\n")
        file.write("\t".join(["filters", str(args.filters)]))
        file.write("\n")
        file.write("\t".join(["n_dil_layers", str(args.n_dilation_layers)]))
        file.write("\n")
        file.write("\t".join(["inputlen", str(args.inputlen)]))
        file.write("\n")
        file.write("\t".join(["outputlen", str(args.outputlen)]))
        file.write("\n")
        file.write("\t".join(["max_jitter", str(args.max_jitter)]))
        file.write("\n")
        file.write("\t".join(["chr_fold_path", str(args.chr_fold_path)]))
        file.write("\n")
        file.write("\t".join(["negative_sampling_ratio",
                              str(1.0)]))  # this is just a dummy variable because the train.py pipeline needs it - all negatives will be used for bias model training
        file.close()


if __name__=="__main__":
    # read the arguments
    parser = parse_data_args()
    args = parse_model_args(parser)

    main(args)
