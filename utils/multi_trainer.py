import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from utils.dataset import SpeechDataset, charset, HandwritingDataset
from utils.multi_model import Encoder_Decoder_Multi
import os
import numpy as np
from utils.criterion import InfoNCE
from tqdm import trange
from utils.sample_positive_negative import get_batch
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List
from utils.load_model_states import save_checkpoint, load_checkpoint
from utils.trainer import get_dataset_loaders_speech_nejm

adv_eps_s = {'speech': 11.343223258948479,
            'nlp10': 5.501777556548058,
            'nlp21': 2.1584105470804023}

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

    B = len(X)
    feat_dim = X[0].shape[-1]
    max_T = max(x.shape[0] for x in X)

    X_padded = torch.zeros(B, max_T, feat_dim, dtype=X[0].dtype)

    for i, x in enumerate(X):
        T = x.shape[0]
        X_padded[i, :T] = x
        if T < max_T:
            X_padded[i, T:] = x[-1:]  

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

    train_loader = DataLoader(train_ds, batch_size=batchSize, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=_padding,
                              persistent_workers=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData



def get_dataset_loaders_nlp_10(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    final_day = 5
    train_input = get_10_input(dataset_name, norm=True ,train=True, days=range(final_day), gauss=not gauss_in, gauss_sigma=2.0)
    test_input =  get_10_input(dataset_name, norm=True ,train=False, days=range(10), valid=True, gauss=not gauss_in, gauss_sigma=2.0)

    valid_set = HandwritingDataset(test_input)
    train_set = HandwritingDataset(train_input)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, None


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
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        valid=True,
        gauss_sigma=2.0
    )
    valid_input_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/recalibration/mat/"),
        norm=True,
        valid=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input = valid_input_1 + valid_input_2
    valid_set = HandwritingDataset(valid_input)
    train_set = HandwritingDataset(train_input)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, None

def get_dataset_loaders(
        speech_dataset,
        nlp10_dataset,
        nlp21_dataset,
        nejm_dataset,
        batch_size, 
        gauss_in=True, 
    ):
    speech_data = get_dataset_loaders_speech(speech_dataset, batch_size, gauss_in)
    nlp21_data = get_dataset_loaders_nlp_21(nlp21_dataset, batch_size, gauss_in)
    nlp10_data = get_dataset_loaders_nlp_10(nlp10_dataset, batch_size, gauss_in)
    nejm_data = get_dataset_loaders_speech_nejm(nejm_dataset, batch_size, gauss_in)
    all_loaders = [
        ('speech', 256, 41, speech_data),
        ('nlp10', 192, 32, nlp10_data),
        ('nlp21', 192, 32, nlp21_data),
        ('nejm', 512, 41, nejm_data)
    ]
    return all_loaders



def train_model(args):
    device = "cuda"
    adv_norm = args.get('adv_norm', 'linf')
    checkpoint_address = args["out_dir"] + "/checkpoint.pt"
    no_noise = args.get('no_noise', False)
    adv_eps_on_dataset = args.get("diff_eps", False)
    adv = args.get('adv', False)
    adv_epsilon = args.get('adv_eps', 0.01)
    model = Encoder_Decoder_Multi(
        args['ceb_out'],
        args['kernel'],
        args['stride'],
        args['hidden'],
        args['layers'],
        args['dropout'],
        args['bidir'],
        args['cebra_unfolder'],
        args['gru'],
        2.0,
        gauss_in=args.get("gauss_in", True),
        shared_rnn=args.get('shared_rnn', False)
    )
    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    criterion = InfoNCE(args['temperature'])
    with open(args["out_dir"] + "/args", "wb") as file:
        pickle.dump(args, file)
    all_loaders = get_dataset_loaders(args['speech_data_dir'], args['nlp_10_data_dir'], args['nlp_21_data_dir'], args['nejm_dataset'], 
                                      args["batchSize"], args.get("gauss_in", True),)
    train_loaders = []
    test_loaders = []
    for session_name, neural_dim, out_dim, (train_loader, test_loader, loaded_data) in all_loaders:
        model.add_session(session_name, neural_dim, out_dim)
        train_loaders.append((session_name, train_loader))
        test_loaders.append((session_name, test_loader))
    
    train_iters = [iter(loader) for session_name, loader in train_loaders]


    model = model.to(device)
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
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

    so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
    testLoss = []
    testCER = []
    inf_losses = 0
    for batch in trange(args["nBatch"]):
        model.train()
        
        for i, (session_name, train_loader) in enumerate(train_loaders):
            
            train_iter = train_iters[i]
            try:
                X, y, X_len, y_len, dayIdx = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_iters[i] = train_iter
                X, y, X_len, y_len, dayIdx = next(train_iter)
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                dayIdx.to(device),
            )
            if batch < so_far_batch:
                continue
            
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
                pred, lengths = model(X, X_len, session_name)
                embeddings, emb_lengths = model.get_cebra_embs()
                ctc_loss = ctc_criterion(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    y,
                    lengths,
                    y_len,
                )
                ctc_loss = torch.sum(ctc_loss)
                reference, positive, negative, ref_batch_idx, ref_time_idx, pos_time_idx, neg_batch_idx, neg_time_idx, = get_batch(embeddings, emb_lengths, args['cont_batch'], args['offset'], False, args.get('random_offset', False), True, False)
                loss_contrastive = criterion(reference, positive, negative)[0]
                loss = loss_contrastive + ctc_loss
                # Backpropagation
            
            if not torch.isfinite(loss):
                inf_losses += 1
                if inf_losses > 10:
                    break
            
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if adv:
                epsilon = adv_eps_s[session_name] / 2 if adv_eps_on_dataset else adv_epsilon
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
                        pred_adv, lengths = model(X_adv, X_len, session_name)
                        if isinstance(model, torch.nn.DataParallel):
                            embeddings_adv, emb_lengths = model.module.get_cebra_embs()
                        else:
                            embeddings_adv, emb_lengths = model.get_cebra_embs()
                        ctc_loss_adv = ctc_criterion(
                            torch.permute(pred_adv.log_softmax(2), [1, 0, 2]),
                            y,
                            lengths,
                            y_len,
                        )
                        ctc_loss_adv = torch.sum(ctc_loss_adv)
                        reference, positive, negative = embeddings_adv[ref_batch_idx, ref_time_idx], embeddings_adv[ref_batch_idx, pos_time_idx], embeddings_adv[neg_batch_idx, neg_time_idx]
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
                    
                    pred_adv, lengths = model(X_adv, X_len, session_name)
                    if isinstance(model, torch.nn.DataParallel):
                        embeddings_adv, emb_lengths = model.module.get_cebra_embs()
                    else:
                        embeddings_adv, emb_lengths = model.get_cebra_embs()
                    ctc_loss_adv = ctc_criterion(
                            torch.permute(pred_adv.log_softmax(2), [1, 0, 2]),
                            y,
                            lengths,
                            y_len,
                        )
                    ctc_loss_adv = torch.sum(ctc_loss_adv)
                    reference, positive, negative = embeddings_adv[ref_batch_idx, ref_time_idx], embeddings_adv[ref_batch_idx, pos_time_idx], embeddings_adv[neg_batch_idx, neg_time_idx]
                    loss_contrastive_adv = criterion(reference, positive, negative)[0]
                    loss_adv = loss_contrastive_adv + ctc_loss_adv
                
                if not torch.isfinite(loss_adv):
                    inf_losses += 1
                    if inf_losses > 10:
                        break
                loss_adv.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            

        
        scheduler.step()


        if batch % 50 == 0:
                model.eval()
                current_iter_cers = {}
                current_iter_losses = {}
                
                
                for session_name, test_loader in test_loaders:
                    with torch.no_grad():
                        total_edit_distance = 0
                        total_seq_length = 0
                        allLoss = []
                        for X, y, X_len, y_len, testDayIdx in test_loader:
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                                X, y, X_len, y_len, testDayIdx = (
                                    X.to(device),
                                    y.to(device),
                                    X_len.to(device),
                                    y_len.to(device),
                                    testDayIdx.to(device),
                                )
                                pred, lengths = model(X, X_len, session_name)
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

                    avg_loss = np.sum(allLoss) / len(test_loader)
                    cer = total_edit_distance / total_seq_length
                    current_iter_cers[session_name] = cer
                    current_iter_losses[session_name] = avg_loss

                avg_cer_sessions = np.mean(list(current_iter_cers.values()))
                for session_name in current_iter_cers:
                    print(
                        f"session {session_name} batch {batch}, ctc loss: {current_iter_losses[session_name]:>f}, cer: {current_iter_cers[session_name]:>7f}"
                    )
                testCER.append(current_iter_cers)
                testLoss.append(current_iter_losses)
                torch.save(model.state_dict(), args["out_dir"] + "/modelWeights")
                save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)
                tStats = {}
                tStats["testLoss"] = (testLoss)
                tStats["testCER"] = (testCER)
                with open(args["out_dir"] + "/trainingStats", "wb") as file:
                    pickle.dump(tStats, file)  



