import os
import random
import numpy as np
import tqdm


def get_sequences(sequence_dir, min_rew=-float('inf'), max_rew=float('inf'), max_ep=None, min_ep_len=1000):
    sequence_files = os.listdir(sequence_dir)
    random.shuffle(sequence_files)
    print(f'Loading episodes from {sequence_dir}')
    print(f'Found {len(sequence_files)} episode files')
    pbar = tqdm.tqdm(total=max_ep if max_ep else len(sequence_files))

    sequences = []
    for episode in sequence_files:
        with open(os.path.join(sequence_dir, episode), 'rb') as f:
            data = np.load(f)
            reward = np.sum(data['reward'])
            if reward < min_rew or reward > max_rew or len(data['reward']) < min_ep_len:
                continue
            data = {k: np.copy(v) for k, v in data.items()}
        seq = []
        failed = False
        for i in range(data['reward'].shape[0]):
            try:
                seq.append({k: v[i] for k, v in data.items()})
            except:
                failed = True
                break
        if failed: continue
        sequences.append(seq)
        pbar.update(1)
        if max_ep is not None and len(sequences) >= max_ep:
            break

    print(f'Found {len(sequences)} episodes with reward in [{min_rew}, {max_rew}]')
    return sequences


def extract_key_from_seq(seq, key):
    return [step[key] for step in seq]


def extract_key_from_seqs(seqs, key, reduce=None):
    all_values_seqs = []
    for seq in seqs:
        values = extract_key_from_seq(seq, key)
        if reduce is not None:
            values = [reduce(values)]
        all_values_seqs.extend(values)
    return all_values_seqs


def construct_im_rows(seqs, key, num_eps, num_ims):
    img_rows = []
    for _ in range(num_eps):
        ep = random.choice(seqs)
        idxs = np.linspace(0, len(ep) - 1, num_ims).astype(np.uint8)
        img_row = []
        for idx in idxs:
            img_row.append(ep[idx][key])
        img_rows.append(np.concatenate(img_row, axis=1))
    img_rows = np.concatenate(img_rows, axis=0)
    return img_rows
