from utils.transformer_trainer import train_model
from tacorn_utils.no_ctc_multi_transformer import Multi_Transformer
import torch

def train_two_phases(args):
    device = torch.device("cuda")
    model = Multi_Transformer(
        args['dim'], args['latent_step'], args['n_latent_step'],
        args['dim_head'], args['n_head'], args.get('shared_rnn', False),
        args['hidden'], args['layers'], args['bidir'], args['dropout'],
        args['stride']
    )
    model.freeze(True)
    args["unfreeze"] = True

    model.add_session_encoder('speech', 256)
    model.add_session_encoder('nlp10', 192)
    model.add_session_encoder('nlp21', 192)
    model.add_session_encoder('nejm', 512)
    model.add_session_decoder('speech', 41)
    model.add_session_decoder('nlp21', 32)
    model.add_session_decoder('nejm', 41)
    model.add_session_decoder('nlp10', 32)
    model.to(device)
    args["nBatch"] = args.get("phase_1_batches", 20000)

    train_model(model, args)
    args["out_dir"] = args["out_dir"] + '_phase2'
    args["unfreeze"] = False
    model.freeze(False)
    args["nBatch"] = args.get("phase_2_batches", 20000)
    train_model(model, args)


