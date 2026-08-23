import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")
from utils.model import Encoder_Decoder
import torch
from utils.eval_utils import get_dataset_loader, eval_model, model_logits

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--datasetPath', type=str, default='/data/hossein/data/speech/speech_data_raw_all_in_test.pkl')

# bools
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--is_nejm', action='store_true', help='eval on nejm dataset')

# Integers
parser.add_argument('--batch_size', type=int, default = 16)


args = parser.parse_args()

with open(args.out_dir + "/args", "rb") as f:
    model_args = pickle.load(f)

is_nlp_10 = args.nlp_10
is_speech = args.is_speech
gauss_in = model_args.get("gauss_in", True)
device = "cuda"

model = Encoder_Decoder(
    (256 if not args.is_nejm else 512) if is_speech else 192,
    model_args['ceb_out'],
    model_args['kernel'],
    model_args['stride'],
    41 if is_speech else 32,
    model_args['hidden'],
    model_args['layers'],
    model_args['dropout'],
    model_args['bidir'],
    model_args['cebra_unfolder'],
    model_args.get('gru', True),
    2.0,
    gauss_in=gauss_in,
    no_rnn=model_args.get("no_rnn", False)
).to(device)
print(model_args["adv"])

model.load_state_dict(torch.load(args.out_dir  + "/modelWeights"))

test_dl = get_dataset_loader(args.datasetPath, args.batch_size, gauss_in, is_speech, is_nlp_10, args.is_nejm)
cer, loss, err_dist = eval_model(model, test_dl)
with open(args.out_dir + "/evalStats.pkl", "wb") as f:
    pickle.dump(err_dist, f)
print(f"CER: {cer:.4f}, Loss: {loss:.4f}")
rnn_outputs = model_logits(model, test_dl, device, not is_speech)
with open(args.out_dir + "/logits", "wb") as f:
    pickle.dump(rnn_outputs, f)

