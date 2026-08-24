import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.model import Encoder_Decoder
from utils.dataset import SpeechDataset, HandwritingDataset

import cebra
import cebra.attribution


# =====================================================================
# helpers
# =====================================================================
def reduce_attr_map(arr):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.abs(np.asarray(arr))
    if arr.ndim == 3:       # (samples, out_dim, in_dim) -> average over samples
        arr = arr.mean(axis=0)
    elif arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float32)


def save_heatmap(arr, path, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Input feature")
    plt.ylabel("Output dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)


def cleanup(*objs):
    import gc
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =====================================================================
# wraps Encoder_Decoder so forward(raw_x) returns the output at a chosen
# intermediate stage -- mirrors Encoder_Decoder.forward exactly, including
# both cebra_unfolder orderings, so it stays correct for any run's config.
# =====================================================================
class EncoderDecoderStageWrapper(nn.Module):
    def __init__(self, encoder_decoder, stage="cebra"):
        super().__init__()
        assert stage in ("cebra", "rnn", "logits")
        self.ed = encoder_decoder
        self.stage = stage

    def forward(self, x):
        ed = self.ed
        lengths = torch.tensor([x.shape[1]] * x.shape[0], device=x.device)

        h = ed.smoother(x)
        if ed.cebra_unfolder:
            h = ed._apply_cebra(h, lengths)
            h, lengths = ed.unfolder(h, lengths)
        else:
            h, lengths = ed.unfolder(h, lengths)
            h = ed._apply_cebra(h, lengths)

        if self.stage == "cebra":
            return h

        h, _ = ed.rnn(h)
        if self.stage == "rnn":
            return h

        h = ed.final_decoder(h)
        return h


# =====================================================================
# rebuild the exact architecture + load trained weights
# =====================================================================
def load_trained_model(out_dir, device):
    with open(os.path.join(out_dir, "args"), "rb") as f:
        model_args = pickle.load(f)

    is_speech = model_args.get("is_speech", True)
    is_nejm = model_args.get("is_nejm", False)
    neural_dim = (256 if not is_nejm else 512) if is_speech else 192
    num_classes = 41 if is_speech else 32

    model = Encoder_Decoder(
        neural_dim,
        model_args["ceb_out"],
        model_args["kernel"],
        model_args["stride"],
        num_classes,
        model_args["hidden"],
        model_args["layers"],
        model_args["dropout"],
        model_args["bidir"],
        model_args["cebra_unfolder"],
        model_args.get("gru", True),
        2.0,
        gauss_in=model_args.get("gauss_in", True),
        no_rnn=model_args.get("no_rnn", False),
        cebra_bn=model_args.get("ceb_bn", False),
        cebra_window_10=model_args.get("cebra_window_10", False),
    ).to(device)

    state = torch.load(os.path.join(out_dir, "modelWeights"), map_location=device)
    model.load_state_dict(state)
    model.eval()  # disable dropout so attribution is deterministic
    print(f"loaded model from {out_dir} | neural_dim={neural_dim} | num_classes={num_classes} "
          f"| ceb_out={model_args['ceb_out']} | hidden={model_args['hidden']} | bidir={model_args['bidir']} "
          f"| cebra_unfolder={model_args['cebra_unfolder']} | no_rnn={model_args.get('no_rnn', False)}")
    return model, model_args


# =====================================================================
# pull ONE real, already-preprocessed trial from the same test split
# trainer.py used, so smoothing/etc. stay consistent with training
# =====================================================================
def day_trial_to_flat_index(test_days, day_idx, trial_in_day):
    """SpeechDataset flattens all days' trials into one list, in order
    (day0-trial0, day0-trial1, ..., day1-trial0, ...). This converts a
    human-readable (day_idx, trial_in_day) into that flat index."""
    if day_idx >= len(test_days):
        raise ValueError(f"day_idx={day_idx} out of range, test split has {len(test_days)} days")
    n_trials_this_day = len(test_days[day_idx]["sentenceDat"])
    if trial_in_day >= n_trials_this_day:
        raise ValueError(f"trial_in_day={trial_in_day} out of range, "
                          f"day {day_idx} only has {n_trials_this_day} trials")
    offset = sum(len(test_days[d]["sentenceDat"]) for d in range(day_idx))
    return offset + trial_in_day


def get_reference_trial(model_args, device, trial_index=None, day_idx=None, trial_in_day=None):
    with open(model_args["datasetPath"], "rb") as f:
        loadedData = pickle.load(f)

    gauss_in = model_args.get("gauss_in", True)
    is_speech = model_args.get("is_speech", True)

    if is_speech:
        # gauss=not gauss_in: avoids double-smoothing when the model already
        # smooths internally (gauss_in=True), matching trainer.py exactly.
        test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in)
    else:
        test_ds = HandwritingDataset(loadedData["test"])

    if day_idx is not None:
        trial_index = day_trial_to_flat_index(loadedData["test"], day_idx, trial_in_day or 0)
        print(f"day_idx={day_idx}, trial_in_day={trial_in_day or 0} -> flat trial_index={trial_index}")
    elif trial_index is None:
        trial_index = 0

    x, y, x_len, y_len, day = test_ds[trial_index]
    print(f"reference trial: flat_index={trial_index} (belongs to day={day})")
    return x.unsqueeze(0).to(device)  # (1, T, F)


# =====================================================================
# run attribution for one stage
# =====================================================================
def run_attribution_for_stage(model, x_ref, stage, out_dim, out_dir, device, tag):
    wrapper = EncoderDecoderStageWrapper(model, stage=stage).to(device)
    wrapper.eval()

    x_tensor = x_ref.clone().detach().to(device)
    x_tensor.requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=wrapper,
        input_data=x_tensor,
        output_dimension=out_dim,
    )
    # batch_size = number of SAMPLES (x_tensor.shape[0]), not sequence length
    result = method.compute_attribution_map(batch_size=x_tensor.shape[0])

    jf = result["jf"]
    jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq", result.get("jf-inv")))

    torch.save(jf, os.path.join(out_dir, f"{tag}_{stage}_jf.pt"))
    torch.save(jf_inv, os.path.join(out_dir, f"{tag}_{stage}_jf_inv.pt"))

    save_heatmap(reduce_attr_map(jf), os.path.join(out_dir, f"{tag}_{stage}_jf.png"),
                 f"{tag} [{stage}] - Jacobian")
    save_heatmap(reduce_attr_map(jf_inv), os.path.join(out_dir, f"{tag}_{stage}_jf_inv.png"),
                 f"{tag} [{stage}] - inverse Jacobian")

    cleanup(wrapper, x_tensor, method, result)


# =====================================================================
# main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="same out_dir used for start_trainer.py")
    parser.add_argument("--stages", nargs="+", default=["cebra", "rnn", "logits"],
                         choices=["cebra", "rnn", "logits"])
    parser.add_argument("--trial_index", type=int, default=None,
                         help="flat index into the whole test split (all days concatenated). "
                              "Ignored if --day_idx is given.")
    parser.add_argument("--day_idx", type=int, default=None,
                         help="pick a specific test-split day (0-based) instead of a flat index")
    parser.add_argument("--trial_in_day", type=int, default=0,
                         help="trial index within --day_idx (default: first trial of that day)")
    parser.add_argument("--tag", type=str, default="attr")
    cli_args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_args = load_trained_model(cli_args.out_dir, device)
    x_ref = get_reference_trial(model_args, device, cli_args.trial_index,
                                 cli_args.day_idx, cli_args.trial_in_day)
    print(f"reference trial shape: {tuple(x_ref.shape)}")

    stages = list(cli_args.stages)
    if model_args.get("no_rnn", False) and "rnn" in stages:
        print("no_rnn=True for this run -- 'rnn' stage doesn't exist, skipping it")
        stages = [s for s in stages if s != "rnn"]

    stage_dims = {
        "cebra": model_args["ceb_out"],
        "rnn": model_args["hidden"] * (2 if model_args["bidir"] else 1),
        "logits": 41 if model_args.get("is_speech", True) else 32,
    }

    for stage in stages:
        print(f"\n=== attribution: stage={stage} | output_dim={stage_dims[stage]} ===")
        run_attribution_for_stage(model, x_ref, stage, stage_dims[stage],
                                   cli_args.out_dir, device, tag=cli_args.tag)

    print("\nDONE")
    


# import os
# import pickle
# import argparse

# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from utils.model import Encoder_Decoder
# from utils.dataset import SpeechDataset, HandwritingDataset

# import cebra
# import cebra.attribution


# # =====================================================================
# # helpers
# # =====================================================================
# def reduce_attr_map(arr):
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     arr = np.abs(np.asarray(arr))
#     if arr.ndim == 3:       # (samples, out_dim, in_dim) -> average over samples
#         arr = arr.mean(axis=0)
#     elif arr.ndim == 1:
#         arr = arr[None, :]
#     return arr.astype(np.float32)


# def save_heatmap(arr, path, title):
#     plt.figure(figsize=(10, 6))
#     plt.imshow(arr, aspect="auto", cmap="viridis")
#     plt.colorbar(label="absolute attribution")
#     plt.xlabel("Input feature")
#     plt.ylabel("Output dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("saved:", path)


# def cleanup(*objs):
#     import gc
#     for o in objs:
#         del o
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# # =====================================================================
# # wraps Encoder_Decoder so forward(raw_x) returns the output at a chosen
# # intermediate stage -- mirrors Encoder_Decoder.forward exactly, including
# # both cebra_unfolder orderings, so it stays correct for any run's config.
# # =====================================================================
# class EncoderDecoderStageWrapper(nn.Module):
#     def __init__(self, encoder_decoder, stage="cebra"):
#         super().__init__()
#         assert stage in ("cebra", "rnn", "logits")
#         self.ed = encoder_decoder
#         self.stage = stage

#     def forward(self, x):
#         ed = self.ed
#         lengths = torch.tensor([x.shape[1]] * x.shape[0], device=x.device)

#         h = ed.smoother(x)
#         if ed.cebra_unfolder:
#             h = ed._apply_cebra(h, lengths)
#             h, lengths = ed.unfolder(h, lengths)
#         else:
#             h, lengths = ed.unfolder(h, lengths)
#             h = ed._apply_cebra(h, lengths)

#         if self.stage == "cebra":
#             return h

#         h, _ = ed.rnn(h)
#         if self.stage == "rnn":
#             return h

#         h = ed.final_decoder(h)
#         return h


# # =====================================================================
# # rebuild the exact architecture + load trained weights
# # =====================================================================
# def load_trained_model(out_dir, device):
#     with open(os.path.join(out_dir, "args"), "rb") as f:
#         model_args = pickle.load(f)

#     is_speech = model_args.get("is_speech", True)
#     is_nejm = model_args.get("is_nejm", False)
#     neural_dim = (256 if not is_nejm else 512) if is_speech else 192
#     num_classes = 41 if is_speech else 32

#     model = Encoder_Decoder(
#         neural_dim,
#         model_args["ceb_out"],
#         model_args["kernel"],
#         model_args["stride"],
#         num_classes,
#         model_args["hidden"],
#         model_args["layers"],
#         model_args["dropout"],
#         model_args["bidir"],
#         model_args["cebra_unfolder"],
#         model_args.get("gru", True),
#         2.0,
#         gauss_in=model_args.get("gauss_in", True),
#         no_rnn=model_args.get("no_rnn", False),
#         cebra_bn=model_args.get("ceb_bn", False),
#         cebra_window_10=model_args.get("cebra_window_10", False),
#     ).to(device)

#     state = torch.load(os.path.join(out_dir, "modelWeights"), map_location=device)
#     model.load_state_dict(state)
#     model.eval()  # disable dropout so attribution is deterministic
#     print(f"loaded model from {out_dir} | neural_dim={neural_dim} | num_classes={num_classes} "
#           f"| ceb_out={model_args['ceb_out']} | hidden={model_args['hidden']} | bidir={model_args['bidir']} "
#           f"| cebra_unfolder={model_args['cebra_unfolder']} | no_rnn={model_args.get('no_rnn', False)}")
#     return model, model_args


# # =====================================================================
# # pull ONE real, already-preprocessed trial from the same test split
# # trainer.py used, so smoothing/etc. stay consistent with training
# # =====================================================================
# def get_reference_trial(model_args, device, trial_index=0):
#     with open(model_args["datasetPath"], "rb") as f:
#         loadedData = pickle.load(f)

#     gauss_in = model_args.get("gauss_in", True)
#     is_speech = model_args.get("is_speech", True)

#     if is_speech:
#         # gauss=not gauss_in: avoids double-smoothing when the model already
#         # smooths internally (gauss_in=True), matching trainer.py exactly.
#         test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in)
#     else:
#         test_ds = HandwritingDataset(loadedData["test"])

#     x, y, x_len, y_len, day = test_ds[trial_index]
#     return x.unsqueeze(0).to(device)  # (1, T, F)


# # =====================================================================
# # run attribution for one stage
# # =====================================================================
# def run_attribution_for_stage(model, x_ref, stage, out_dim, out_dir, device, tag):
#     wrapper = EncoderDecoderStageWrapper(model, stage=stage).to(device)
#     wrapper.eval()

#     x_tensor = x_ref.clone().detach().to(device)
#     x_tensor.requires_grad_(True)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=wrapper,
#         input_data=x_tensor,
#         output_dimension=out_dim,
#     )
#     # batch_size = number of SAMPLES (x_tensor.shape[0]), not sequence length
#     # result = method.compute_attribution_map(batch_size=x_tensor.shape[0])
#     with torch.backends.cudnn.flags(enabled=False):
#         result = method.compute_attribution_map(
#             batch_size=x_tensor.shape[0]
#         )

#     jf = result["jf"]
#     jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq", result.get("jf-inv")))

#     torch.save(jf, os.path.join(out_dir, f"{tag}_{stage}_jf.pt"))
#     torch.save(jf_inv, os.path.join(out_dir, f"{tag}_{stage}_jf_inv.pt"))

#     save_heatmap(reduce_attr_map(jf), os.path.join(out_dir, f"{tag}_{stage}_jf.png"),
#                  f"{tag} [{stage}] - Jacobian")
#     save_heatmap(reduce_attr_map(jf_inv), os.path.join(out_dir, f"{tag}_{stage}_jf_inv.png"),
#                  f"{tag} [{stage}] - inverse Jacobian")

#     cleanup(wrapper, x_tensor, method, result)


# # =====================================================================
# # main
# # =====================================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_dir", required=True, help="same out_dir used for start_trainer.py")
#     parser.add_argument("--stages", nargs="+", default=["cebra", "rnn", "logits"],
#                          choices=["cebra", "rnn", "logits"])
#     parser.add_argument("--trial_index", type=int, default=0,
#                          help="which test-set trial to use as the attribution reference")
#     parser.add_argument("--tag", type=str, default="attr")
#     cli_args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model, model_args = load_trained_model(cli_args.out_dir, device)
#     x_ref = get_reference_trial(model_args, device, cli_args.trial_index)
#     print(f"reference trial shape: {tuple(x_ref.shape)}")

#     stages = list(cli_args.stages)
#     if model_args.get("no_rnn", False) and "rnn" in stages:
#         print("no_rnn=True for this run -- 'rnn' stage doesn't exist, skipping it")
#         stages = [s for s in stages if s != "rnn"]

#     stage_dims = {
#         "cebra": model_args["ceb_out"],
#         "rnn": model_args["hidden"] * (2 if model_args["bidir"] else 1),
#         "logits": 41 if model_args.get("is_speech", True) else 32,
#     }

#     for stage in stages:
#         print(f"\n=== attribution: stage={stage} | output_dim={stage_dims[stage]} ===")
#         run_attribution_for_stage(model, x_ref, stage, stage_dims[stage],
#                                    cli_args.out_dir, device, tag=cli_args.tag)

#     print("\nDONE")
