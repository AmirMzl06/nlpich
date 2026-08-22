from utils.loaders_transformer import get_dataset_loaders
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

def train_model(model, args):
    device = "cuda"
    adv_norm = args.get('adv_norm', 'linf')
    unfreeze = args.get("unfreeze", False)
    checkpoint_address = args["out_dir"] + f"/checkpoint{'_unf' if unfreeze else '_fr'}.pt"
    adv = args.get('adv', False)
    adv_epsilon = args.get('adv_eps', 0.01)
    model.freeze(unfreeze)

    model.to(device)
    so_far_batch = 0
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
    else:
        encoder_params = []
        other_params = []

        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)

        optimizer_main = torch.optim.Adam(other_params, lr=args["lrStart"],
                                          betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"], )
        optimizer_encoder = torch.optim.Adam(encoder_params, lr=args["lrStart"],
                                             betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"], )
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




    model.eval()

    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["out_dir"] + "/args", "wb") as file:
        pickle.dump(args, file)



    all_loaders = get_dataset_loaders(
        args['speech_data_dir'], args['nlp_10_data_dir'], args['nlp_21_data_dir'],
        args['nejm_dataset'], args["batchSize"], True, encoder=None if unfreeze else model
    )

    train_loaders = []
    test_loaders = []
    for session_name, decoder_name, neural_dim, out_dim, (train_loader, test_loader, loaded_data) in all_loaders:
        decoder_name = session_name
        train_loaders.append((session_name, decoder_name, train_loader))
        test_loaders.append((session_name, decoder_name, test_loader))

    train_iters = [InfiniteDataLoader(loader) for session_name, dec_name, loader in train_loaders]
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    testLoss = []
    testCER = []
    inf_losses = 0

    for batch in trange(args["nBatch"]):
        freeze_iter = (batch // 500) % 3 == 0
        if not unfreeze:
            optimizer, scheduler = optimizers[0], schedulers[0]
        else:
            for p in model.parameters():
                p.requires_grad_(not freeze_iter)

            for p in model.encoder.parameters():
                p.requires_grad_(freeze_iter)
            idx_opt = 0 if freeze_iter else 1
            scheduler = schedulers[idx_opt]
            optimizer = optimizers[idx_opt]


        if batch < so_far_batch:
            continue
        model.train()

        for i, (session_name, decoder_name, train_loader) in enumerate(train_loaders):
            X, y, X_len, y_len, dayIdx = train_iters[i].next()

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

            for session_name, decoder_name, test_loader in test_loaders:
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