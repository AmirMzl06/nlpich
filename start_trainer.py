from utils.trainer import train_model
import argparse

parser = argparse.ArgumentParser(description="Train Neural Decoder")

# Strings
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--datasetPath', type=str, default='/data/hossein/data/speech/speech_data_raw.npz')
parser.add_argument('--adv_norm', type=str, default='linf')

# Booleans (Actions are inverse to their defaults)
parser.add_argument('--adv', action='store_true', help='adversarially training')
parser.add_argument('--no_noise', action='store_true', help= 'no noise added')
parser.add_argument('--bidir',  action='store_true', help='bidirectional rnn')
parser.add_argument('--cebra_unfolder',  action='store_true', help='cebra before unfolder')
parser.add_argument('--gru',  action='store_true', help='gru instead of lstm')
parser.add_argument('--gauss_in', action='store_true', help='gaussian kernel inside model')
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--sample_single', action='store_true', help='sample ref steps from 1 trial')
parser.add_argument('--no_rnn', action='store_true', help="linear decoder")
parser.add_argument('--all_ref', action='store_true', help='all times are ref')
parser.add_argument('--random_dir', action='store_true', help='random window direction')
parser.add_argument('--cebra_window_10', action='store_true', help='cebra 10 model')
parser.add_argument('--ceb_bn', action='store_true', help='cebra bn model')
parser.add_argument('--no_gauss', action='store_true', help='no gaussian smoothing at all')
parser.add_argument('--is_nejm', action='store_true', help='nejm speech')
parser.add_argument('--random_offset', action='store_true', help='offset dynamic from 1 to <offet>')
# Integers
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--nBatch', type=int, default=50000)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden', type=int, default=1024)
parser.add_argument('--nInputFeatures', type=int, default=256)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--kernel', type=int, default=32)
parser.add_argument('--cebra_model', type=int, default=36)
parser.add_argument('--cont_batch', type=int, default=1024)
parser.add_argument('--offset', type=int, default=1)
parser.add_argument('--ceb_out', type=int, default=64)

# Floats
parser.add_argument('--lrStart', type=float, default=0.02)
parser.add_argument('--lrEnd', type=float, default=0.002)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--temperature', type=float, default=0.4)
parser.add_argument('--whiteNoiseSD', type=float, default=0.8)
parser.add_argument('--constantOffsetSD', type=float, default=0.2)
parser.add_argument('--l2_decay', type=float, default=1e-5)
parser.add_argument('--adv_eps', type=float, default=0.01)

# Parse arguments
parsed_args = parser.parse_args()

# Convert namespace to dictionary
args_dict = vars(parsed_args)
train_model(args_dict)
