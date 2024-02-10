import chrombpnet.training.data_generators.batchgen_generator as batchgen_generator
from chrombpnet.training.utils import data_utils
import pandas as pd
import json

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]

def fetch_data_and_model_params_based_on_mode(mode, args, parameters):
    if mode == "train":
        inputlen = int(parameters["inputlen"])
        outputlen = int(parameters["outputlen"])
        max_jitter = int(parameters["max_jitter"])
        #add_revcomp = True
        add_revcomp = False
        shuffle_at_epoch_start = True

    elif mode == "valid":
        inputlen = int(parameters["inputlen"])
        outputlen = int(parameters["outputlen"])

        # do not jitter at valid time - we are testing only at summits
        max_jitter = 0
        # no reverse complementation at valid time
        add_revcomp = False
        # no need to shuffle
        shuffle_at_epoch_start = False

    elif mode == "test":
        # read input/output length
        inputlen = args.inputlen
        outputlen = args.outputlen

        # no jitter at valid time - we are testing only at summits
        max_jitter = 0
        # no reverse complementation at test time
        add_revcomp = False
        # no need to shuffle
        shuffle_at_epoch_start = False

    else:
        print("mode not defined - only train, valid, test are allowed")

    return inputlen, outputlen, max_jitter, add_revcomp, shuffle_at_epoch_start

def get_bed_regions_for_fold_split(bed_regions, mode, splits_dict):
    chroms_to_keep=splits_dict[mode]
    bed_regions_to_keep=bed_regions[bed_regions["chr"].isin(chroms_to_keep)]
    print("got split:"+str(mode)+" for bed regions:"+str(bed_regions_to_keep.shape))
    return bed_regions_to_keep, chroms_to_keep

def initialize_generators(args, mode, parameters, return_coords):
    # defaults
    peak_regions = None

    # get only those peak/non peak regions corresponding to train/valid/test set
    splits_dict = json.load(open(args.chr_fold_path))

    if args.peaks.lower() != "none":
        print("loading peaks...")
        peak_regions = pd.read_csv(args.peaks, header=None, sep='\t', names=NARROWPEAK_SCHEMA)
        peak_regions, chroms = get_bed_regions_for_fold_split(peak_regions, mode, splits_dict)

    inputlen, outputlen, \
    max_jitter, add_revcomp, shuffle_at_epoch_start = fetch_data_and_model_params_based_on_mode(mode, args, parameters)

    generator = batchgen_generator.ChromBPNetBatchGenerator(
        peak_regions=peak_regions,
        genome_fasta=args.genome,
        batch_size=args.batch_size,
        inputlen=inputlen,
        outputlen=outputlen,
        max_jitter=max_jitter,
        cts_bw_file=args.bigwig,
        add_revcomp=add_revcomp,
        return_coords=return_coords,
        shuffle_at_epoch_start=shuffle_at_epoch_start
    )

    return generator
