import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_utils import (
    cleanup_ddp,
    gather_with_grad,
    get_rank,
    is_distributed,
    gather_with_grad_seq,
    is_main_process,
    setup_ddp,
    get_sampler,
    get_world_size
)
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from utils.dataset import SpeechDataset, charset, HandwritingDataset
from utils.model import Encoder_Decoder
import os
import numpy as np
from utils.criterion import InfoNCE
from tqdm import trange
from utils.sample_positive_negative import get_batch
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List




def ctc_collate(batch: List[Tuple[torch.Tensor, str, int]]):
    """
    Returns:
      x_pad: (B, T_max, F)
      input_lengths: (B,)
      targets: (sum_L,)
      target_lengths: (B,)
      transcripts: list[str]
      sessions: (B,)
    """
    xs, ys, ds = zip(*batch)

    B = len(xs)
    feat_dim = xs[0].shape[-1]

    input_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T_max = int(input_lengths.max().item())

    x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        x_pad[i, T:] = x[-1:]

    target_seqs = [torch.tensor(charset.text_to_int(y), dtype=torch.long) for y in ys]
    target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    targets = torch.cat(target_seqs) if len(target_seqs) else torch.tensor([], dtype=torch.long)
    max_target_len = max(target_lengths) if len(target_lengths) > 0 else 0
    targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)

    offset = 0
    for i, length in enumerate(target_lengths):
        targets_padded[i, :length] = targets[offset:offset + length]
        offset += length


    sessions = torch.tensor(ds, dtype=torch.long)

    return x_pad, targets_padded, input_lengths, target_lengths, sessions





def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )


def get_dataset_loaders_speech(
        datasetName,
        batchSize,
        gauss_in=False
    ):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    


    train_ds = SpeechDataset(loadedData["train"], transform=None, gauss=not gauss_in)
    test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in)
    sampler = get_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batchSize,
                              sampler=sampler,
                              num_workers=4, pin_memory=True, collate_fn=_padding,
                              persistent_workers=True, drop_last=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, sampler



def get_dataset_loaders_nlp_10(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    final_day = 6
    train_input = get_10_input(dataset_name, norm=True ,train=True, days=range(final_day), gauss=not gauss_in, gauss_sigma=2.0)
    test_input =  get_10_input(dataset_name, norm=True ,train=False, days=range(final_day, 10), gauss=not gauss_in, gauss_sigma=2.0)

    valid_set = HandwritingDataset(test_input)
    train_set = HandwritingDataset(train_input)
    
    sampler = get_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True, drop_last=True)

    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, sampler


def get_dataset_loaders_nlp_21(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    train_input = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=True, 
        gauss_sigma=2.0
    )
    valid_input_1 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/no_recalibration/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/no_recalibration/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input = valid_input_1 + valid_input_2
    valid_set = HandwritingDataset(valid_input)
    train_set = HandwritingDataset(train_input)
    sampler = get_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True, drop_last=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, sampler

def get_dataset_loaders(
        dataset_name,
        batch_size, 
        gauss_in=True, 
        speech=True,
        nlp_10=False
    ):
    if speech:
        return get_dataset_loaders_speech(dataset_name, batch_size, gauss_in)
    if not nlp_10:
        return get_dataset_loaders_nlp_21(dataset_name, batch_size, gauss_in)
    return get_dataset_loaders_nlp_10(dataset_name, batch_size, gauss_in)



def train_model(args : dict):
    
    
    is_speech = args.get("is_speech", True)
    adv_norm = args.get('adv_norm', 'linf')
    sample_single = args.get("sample_single", False)
    no_noise = args.get('no_noise', False)
    adv = args.get('adv', False)
    adv_eps = args.get('adv_eps', 0.01)
    no_rnn = args.get("no_rnn", False)
    local_rank = setup_ddp()
    world_size = get_world_size()

    seed = args.get("seed", 0) + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = Encoder_Decoder(
        256 if is_speech else 192, 
        args['ceb_out'],
        args['kernel'],
        args['stride'],
        41 if is_speech else 32,
        args['hidden'],
        args['layers'],
        args['dropout'],
        args['bidir'],
        args['cebra_unfolder'],
        args['gru'],
        2.0,
        gauss_in=args.get("gauss_in", True),
        no_rnn=no_rnn
    ).to(device)
    
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if is_main_process():
        os.makedirs(args["out_dir"], exist_ok=True)
    
    
    criterion = InfoNCE(args['temperature'])
    with open(args["out_dir"] + "/args", "wb") as file:
        pickle.dump(args, file)
    trainLoader, testLoader, sampler = get_dataset_loaders(
        args["datasetPath"],
        args["batchSize"] // world_size,
        args.get("gauss_in", True),
        is_speech,
        args.get("nlp_10", False)
    )
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        ddp_model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )
    inf_losses = 0
    cont_batch = args['cont_batch'] // world_size
    
    testLoss = []
    testCER = []
    train_iter = iter(trainLoader)
    curr_epoch = 0
    steps_per_epoch = len(trainLoader)
    sampler.set_epoch(curr_epoch)
    for batch in trange(args["nBatch"], disable=not is_main_process()):
        ddp_model.train()
        if batch > 0 and batch % steps_per_epoch == 0:
            curr_epoch += 1
            sampler.set_epoch(curr_epoch)
            train_iter = iter(trainLoader)

        X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )
        if not no_noise:
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                        torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                        * args["constantOffsetSD"]
                    )
    
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # Clean Forward
            pred, lengths = ddp_model(X, X_len)
            
            embeddings, emb_lengths = ddp_model.module.get_cebra_embs()
            ctc_loss = ctc_criterion(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                lengths,
                y_len,
            )
            ctc_loss = torch.sum(ctc_loss)
            reference, positive, negative, ref_batch_idx, ref_time_idx, pos_time_idx, neg_batch_idx, neg_time_idx, = get_batch(embeddings, emb_lengths, cont_batch, args['offset'], sample_single)
            reference, positive, negative = gather_with_grad(reference), gather_with_grad(positive), gather_with_grad(negative)
            loss_contrastive = criterion(reference, positive, negative)[0]
            loss = loss_contrastive + ctc_loss
            # Backpropagation
            optimizer.zero_grad()
        if not torch.isfinite(loss):
            inf_losses += 1
            if inf_losses > 10:
                break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=5.0)
        optimizer.step()

        if adv:
            epsilon = adv_eps
            steps = 10
            alpha = epsilon / 5.0
            
            X_adv = X.detach().clone().to(device)

            if adv_norm == 'linf':
                X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
            elif adv_norm == 'l2':
                noise = torch.randn_like(X_adv)
                noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                noise_normalized = noise / noise_norm
                noise_normalized *= (torch.rand((noise.shape[0], noise.shape[1], 1), device=noise.device) * epsilon)
                X_adv = X_adv + noise_normalized



            for i in range(steps):
                X_adv = X_adv.detach()
                X_adv.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    preds_adv, lengths = ddp_model(X_adv, X_len)
                    embeddings_adv, emb_lengths = ddp_model.module.get_cebra_embs()
                    ctc_loss_adv = ctc_criterion(
                        torch.permute(preds_adv.log_softmax(2), [1, 0, 2]),
                        y,
                        lengths,
                        y_len,
                    )
                    ctc_loss_adv = torch.sum(ctc_loss_adv)
                    reference, positive, negative = embeddings_adv[ref_batch_idx, ref_time_idx], embeddings_adv[ref_batch_idx, pos_time_idx], embeddings_adv[neg_batch_idx, neg_time_idx]
                    reference, positive, negative = gather_with_grad(reference), gather_with_grad(positive), gather_with_grad(negative)
            
                    loss_contrastive_adv = criterion(reference, positive, negative)[0]
                    
                    loss_adv = loss_contrastive_adv + ctc_loss_adv
                
                grad = torch.autograd.grad(loss_adv, X_adv, only_inputs=True)[0]
                
                with torch.no_grad():
                    if adv_norm == 'linf':
                        X_adv = X_adv + alpha * grad.sign()
                        delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
                        X_adv = X + delta
                    elif adv_norm == 'l2':
                        grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                        grad_normalized = grad / grad_norm
                        X_adv = (X_adv + alpha * grad_normalized).detach()
                        delta = X_adv - X
                        delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                        scale = torch.clamp(epsilon / delta_norm, max=1.0)
                        delta = delta * scale
                        X_adv = (X + delta).detach()

            optimizer.zero_grad()
            X_adv = X_adv.detach()
            X_adv.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                
                preds_adv, lengths = ddp_model(X_adv, X_len)
                embeddings_adv, emb_lengths = ddp_model.module.get_cebra_embs()
                ctc_loss_adv = ctc_criterion(
                        torch.permute(preds_adv.log_softmax(2), [1, 0, 2]),
                        y,
                        lengths,
                        y_len,
                    )
                ctc_loss_adv = torch.sum(ctc_loss_adv)
                reference, positive, negative = embeddings_adv[ref_batch_idx, ref_time_idx], embeddings_adv[ref_batch_idx, pos_time_idx], embeddings_adv[neg_batch_idx, neg_time_idx]
                reference, positive, negative = gather_with_grad(reference), gather_with_grad(positive), gather_with_grad(negative)
                loss_contrastive_adv = criterion(reference, positive, negative)[0]
                loss_adv = loss_contrastive_adv + ctc_loss_adv
            
            if not torch.isfinite(loss_adv):
                inf_losses += 1
                if inf_losses > 10:
                    break
            loss_adv.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=5.0)
            optimizer.step()


        
        scheduler.step()
        
        if batch % 50 == 0 and is_main_process():
            
            with torch.no_grad():
                
                ddp_model.module.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:

                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        X, y, X_len, y_len, testDayIdx = (
                            X.to(device),
                            y.to(device),
                            X_len.to(device),
                            y_len.to(device),
                            testDayIdx.to(device),
                        )
                        pred, lengths = ddp_model.module(X, X_len)
                        loss = ctc_criterion(
                            torch.permute(pred.log_softmax(2), [1, 0, 2]),
                            y,
                            lengths,
                            y_len,
                        )
                        loss = torch.sum(loss)
                        allLoss.append(loss.cpu().detach().numpy())

                        
                        for iterIdx in range(pred.shape[0]):
                            decodedSeq = torch.argmax(
                                torch.tensor(pred[iterIdx, 0: lengths[iterIdx], :]),
                                dim=-1,
                            )  # [num_seq,]
                            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                            decodedSeq = decodedSeq.cpu().detach().numpy()
                            decodedSeq = np.array([i for i in decodedSeq if i != 0])

                            trueSeq = np.array(
                                y[iterIdx][0: y_len[iterIdx]].cpu().detach()
                            )
                            matcher = SequenceMatcher(
                                a=trueSeq.tolist(), b=decodedSeq.tolist()
                            )
                            total_edit_distance += matcher.distance()
                            total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, tr_ctc: {loss:>7f}, tr_cont: {loss_contrastive:>7f}"
                )
                ddp_model.module.train()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(ddp_model.module.state_dict(), args["out_dir"] + "/modelWeights")

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["out_dir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

        # dist.barrier()        