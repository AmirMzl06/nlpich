from utils.loaders_transformer import get_dataset_loaders_single
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm, trange
from utils.load_model_states import load_checkpoint, save_checkpoint
from edit_distance import SequenceMatcher
from tacorn_utils.multi_transformer import Multi_Transformer
from utils.sample_positive_negative import get_batch
from utils.criterion import InfoNCE, infonce

def dataset_name(is_speech, is_nejm, nlp10):
    if is_speech:
        return "speech" if not is_nejm else "nejm"
    if nlp10: return "nlp10"
    return "nlp21"

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

def train_model(args):
    device = "cuda"
    adv_norm = args.get('adv_norm', 'linf')
    checkpoint_address = args["out_dir"] + "/checkpoint.pt"
    adv = args.get('adv', False)
    adv_epsilon = args.get('adv_eps', 0.01)
    end_to_end_training = args.get('end_to_end_training', False)
    unfreeze = args.get("unfreeze", False)

    model = Multi_Transformer(
        args['dim'], args['latent_step'], args['n_latent_step'],
        args['dim_head'], args['n_head'], False,
        args['hidden'], args['layers'], args['bidir'], args['dropout'],
        args['kernel'], args['stride']
    )
    model.freeze(unfreeze)
    num_neurons = 192 if not args["is_speech"] else (256 if not args["is_nejm"] else 512)
    decoder_name = 'speech' if args["is_speech"] else "nlp"
    decoder_out = 41 if args["is_speech"] else 32
    session_name = "default"
    model.add_session_encoder(session_name, num_neurons)
    model.add_session_decoder(decoder_name, decoder_out)
    model.to(device)
    so_far_batch = 0
    if not unfreeze:
        pretrain_folder = dataset_name(args["is_speech"], args["is_nejm"], args["nlp_10"]) + "-single"
        if adv:
            pretrain_folder += '-adv'
        print(pretrain_folder)
        model.encoder.load_state_dict(torch.load(f'{pretrain_folder}/pretrained.pt')["model"])
    if not unfreeze:
        for p in model.parameters():
            p.requires_grad_(True)

        for p in model.encoder.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args["lrStart"],
            betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"], total_iters=args["nBatch"],
        )
        optimizers, schedulers = [optimizer], [scheduler]
        so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)

    elif not end_to_end_training:
        encoder_params = []
        other_params = []

        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)

        optimizer_main = torch.optim.Adam(other_params, lr=args["lrStart"],
            betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"],)
        optimizer_encoder = torch.optim.Adam(encoder_params, lr=args["lrStart"],
            betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"],)
        optimizers = [optimizer_main, optimizer_encoder]
        scheduler_main = torch.optim.lr_scheduler.LinearLR(
            optimizer_main, start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"], total_iters=args["nBatch"] * 2 // 3,
        )
        scheduler_encoder = torch.optim.lr_scheduler.LinearLR(
            optimizer_encoder, start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"], total_iters=args["nBatch"] // 3,
        )
        schedulers = [scheduler_main, scheduler_encoder]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lrStart"], betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"], )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,end_factor=args["lrEnd"] / args["lrStart"], total_iters=args["nBatch"])
        optimizers, schedulers = [optimizer], [scheduler]


    model.eval()

    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["out_dir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # pretrain_folder = "multi-transformer-pre-001" if not adv else "multi-transformer-pre-101"


    train_loader, test_loader, _ = get_dataset_loaders_single(args["dataset_path"],16, True, args["is_speech"], args["nlp_10"], args["is_nejm"], encoder=None if unfreeze else model)



    train_iter = InfiniteDataLoader(train_loader)
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    testLoss = []
    testCER = []
    inf_losses = 0

    for batch in trange(args["nBatch"]):
        freeze_iter = (batch // 100) % 5 == 0
        if unfreeze and not end_to_end_training:
            for p in model.parameters():
                p.requires_grad_(not freeze_iter)

            for p in model.encoder.parameters():
                p.requires_grad_(freeze_iter)
            idx_opt = 0 if freeze_iter else 1
            scheduler = schedulers[idx_opt]
            optimizer = optimizers[idx_opt]
        else:
            optimizer, scheduler = optimizers[0], schedulers[0]

        if batch < so_far_batch:
            continue
        else:
            model.train()


            X, y, X_len, y_len, dayIdx = train_iter.next()

            X, y, X_len, y_len = X.to(device), y.to(device), X_len.to(device), y_len.to(device)



            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pred, lengths = model(X, X_len, session_name, decoder_name)
                ctc_loss = ctc_criterion(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    y, lengths, y_len,
                )
                loss = torch.sum(ctc_loss)

            if not torch.isfinite(loss):
                inf_losses += 1
                if inf_losses > 10:
                    break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if adv:
                epsilon = adv_epsilon
                steps = 10
                alpha = epsilon / 5.0

                X_adv = X.detach().clone().to(device)

                if adv_norm == 'linf':
                    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
                elif adv_norm == 'l2':
                    noise = torch.randn_like(X_adv)
                    noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                    X_adv = X_adv + (noise / noise_norm) * (
                                torch.rand((noise.shape[0], noise.shape[1], 1), device=device) * epsilon)

                for _ in range(steps):
                    X_adv = X_adv.detach().requires_grad_(True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        pred_adv, lengths = model(X_adv, X_len, session_name, decoder_name)
                        ctc_loss_adv = torch.sum(ctc_criterion(
                            torch.permute(pred_adv.log_softmax(2), [1, 0, 2]),
                            y, lengths, y_len,
                        ))

                    grad = torch.autograd.grad(ctc_loss_adv, X_adv, only_inputs=True)[0]

                    with torch.no_grad():
                        if adv_norm == 'linf':
                            X_adv = X_adv + alpha * grad.sign()
                            delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
                            X_adv = X + delta
                        elif adv_norm == 'l2':
                            grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                            X_adv = (X_adv + alpha * (grad / grad_norm)).detach()
                            delta = X_adv - X
                            delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                            X_adv = (X + delta * torch.clamp(epsilon / delta_norm, max=1.0)).detach()

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    pred_adv, lengths = model(X_adv.detach(), X_len, session_name, decoder_name)
                    loss_adv = torch.sum(ctc_criterion(
                        torch.permute(pred_adv.log_softmax(2), [1, 0, 2]),
                        y, lengths, y_len,
                    ))

                if not torch.isfinite(loss_adv):
                    inf_losses += 1
                    if inf_losses > 10:
                        break
                loss_adv.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        scheduler.step()

        if batch % 200 == 0:
            model.eval()
            current_iter_cers = {}
            current_iter_losses = {}

            total_edit_distance = 0
            total_seq_length = 0
            allLoss = []

            with torch.no_grad():
                    for X, y, X_len, y_len, testDayIdx in test_loader:
                        X, y, X_len, y_len = X.to(device), y.to(device), X_len.to(device), y_len.to(device)

                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            pred, lengths = model(X, X_len, session_name, decoder_name)
                            loss = torch.sum(ctc_criterion(
                                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                                y, lengths, y_len,
                            ))

                        allLoss.append(loss.item())

                        pred_detached = pred.detach()
                        lengths_cpu = lengths.cpu().numpy()
                        y_cpu = y.cpu().numpy()
                        y_len_cpu = y_len.cpu().numpy()

                        for iterIdx in range(pred_detached.shape[0]):
                            sub_pred = pred_detached[iterIdx, 0: lengths_cpu[iterIdx], :]
                            decodedSeq = torch.argmax(sub_pred, dim=-1)
                            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1).cpu().numpy()
                            decodedSeq = np.array([i for i in decodedSeq if i != 0])

                            trueSeq = y_cpu[iterIdx][0: y_len_cpu[iterIdx]]

                            matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                            total_edit_distance += matcher.distance()
                            total_seq_length += len(trueSeq)

            avg_loss = float(np.sum(allLoss) / len(test_loader))
            cer = float(total_edit_distance / total_seq_length) if total_seq_length > 0 else 0.0
            current_iter_cers[session_name] = cer
            current_iter_losses[session_name] = avg_loss

            for session_name in current_iter_cers:
                print(
                    f"session {session_name} batch {batch}, ctc loss: {current_iter_losses[session_name]:>f}, cer: {current_iter_cers[session_name]:>7f}")

            testCER.append(current_iter_cers)
            testLoss.append(current_iter_losses)

            torch.save(model.state_dict(), args["out_dir"] + "/modelWeights")
            save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)

            tStats = {"testLoss": testLoss, "testCER": testCER}
            with open(args["out_dir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

            torch.cuda.empty_cache()