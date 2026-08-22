import torch

def get_batch(x: torch.Tensor, x_len: torch.Tensor, batch_size: int, offset: int, 
              single_sequence: bool = False, random_offset: bool = False, 
              random_dir: bool = False, all_ref: bool = False):
    B, T, F = x.shape
    device = x.device

    # --- Determine indices ---
    if all_ref:
        # Create a mask of valid indices: shape (B, T)
        # Create a range tensor of shape (1, T)
        time_range = torch.arange(T, device=device).unsqueeze(0)
        # Compare with x_len: shape (B, 1)
        valid_mask = time_range < x_len.unsqueeze(1)
        
        # Get all valid (b, t) pairs
        ref_batch_idx, ref_time_idx = torch.where(valid_mask)
        
        # If single_sequence is enabled with all_ref, pick only one sequence
        if single_sequence:
            # Pick one sequence index randomly
            chosen_batch = torch.randint(0, B, (1,)).item()
            # Filter indices where batch index == chosen_batch
            mask = ref_batch_idx == chosen_batch
            ref_batch_idx = ref_batch_idx[mask]
            ref_time_idx = ref_time_idx[mask]
    else:
        # --- Original random sampling logic ---
        if not single_sequence:
            ref_batch_idx = torch.randint(0, B, (batch_size,), device=device)
            max_times = torch.clamp(x_len[ref_batch_idx] - offset - 1, min=1)
            ref_time_idx = torch.randint(0 if not random_dir else offset, torch.max(max_times).item(), (batch_size,), device=device)
            ref_time_idx = torch.min(ref_time_idx, max_times - 1)
        else:
            pos_batch = torch.randint(0, B, (1,), device=device).item()
            ref_batch_idx = torch.full((batch_size,), pos_batch, device=device)
            max_times = torch.clamp(x_len[pos_batch] - offset - 1, min=1)
            ref_time_idx = torch.randint(0 if not random_dir else offset, max_times.item(), (batch_size,), device=device)

    # --- Calculate offsets and positive indices ---
    if random_offset:
        add_offset = torch.randint(1, offset + 1, (len(ref_batch_idx),), device=device, dtype=torch.long)
    else:
        add_offset = torch.full((len(ref_batch_idx),), offset, device=device, dtype=torch.long)
    
    if random_dir:
        dir_val = torch.randint(0, 2, add_offset.shape, device=device) * 2 - 1
        add_offset = add_offset * dir_val

    # Ensure positive indices don't exceed sequence length
    pos_time_idx = ref_time_idx + add_offset
    # Clamp based on the specific length of the sequences involved
    max_valid = x_len[ref_batch_idx] - 1
    min_val = torch.zeros_like(max_valid) 
    pos_time_idx = torch.clamp(pos_time_idx, min=min_val, max=max_valid)

    # --- Negatives ---
    if single_sequence:
        # Negatives from OTHER sequences
        neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)
        # Simple retry logic to ensure we don't pick the same sequence as the reference
        # (Only if B > 1)
        if B > 1:
            mask = neg_batch_idx == ref_batch_idx[0]
            neg_batch_idx[mask] = (neg_batch_idx[mask] + 1) % B
    else:
        neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)

    neg_max_times = x_len[neg_batch_idx]
    neg_time_idx = torch.randint(0, torch.max(neg_max_times).item(), (len(ref_batch_idx),), device=device)
    neg_time_idx = torch.min(neg_time_idx, neg_max_times - 1)

    # Extract tensors
    reference = x[ref_batch_idx, ref_time_idx]
    positive = x[ref_batch_idx, pos_time_idx]
    negative = x[neg_batch_idx, neg_time_idx]

    return (
        reference,
        positive,
        negative,
        ref_batch_idx,
        ref_time_idx,
        pos_time_idx,
        neg_batch_idx,
        neg_time_idx,
    )
