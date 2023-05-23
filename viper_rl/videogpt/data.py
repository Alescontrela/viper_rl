import os
import os.path as osp
import glob
import io
import math
import random
import pickle
import numpy as np
import jax
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_io as tfio
from flax import jax_utils


def load_dataset(config, train, modality='video'):
    num_data_local = jax.local_device_count()
    num_ds_shards = jax.process_count()
    ds_shard_id = jax.process_index()

    N = num_data_local
    batch_size = config.batch_size // num_ds_shards

    def get_dataset(data_path, data_type, batch_n, initial_shape, mask):
        load_fn = {
            'npz': load_npz,
            'mp4': load_mp4s,
        }[data_type]
        dataset = load_fn(
            config, data_path, train, num_ds_shards, ds_shard_id, mask
        )
        
        if modality == 'image':
            def to_images(features):
                video = features.pop('video')
                assert len(features) == 0, f'Only support video-only dict'
                return tf.data.Dataset.from_tensor_slices(dict(image=video))
            dataset = dataset.flat_map(to_images)

        dataset = prepare(
            dataset, batch_n,
            initial_shape=initial_shape,
        )
        return dataset

    def get_data_type(data_path):
        fns = _glob_files(osp.join(data_path, '*'))
        fns = list(filter(lambda fn: not fn.endswith('pkl'), fns))
        if len(fns) == 2: # 'train' and 'test' directories
            fns = _glob_files(osp.join(data_path, 'test', '*'))
            if 'mp4' in fns[0]:
                return 'mp4', None
            elif 'npz' in fns[0]:
                return 'npz', None
            else:
                raise NotImplementedError(f'Unsupported file type {fns[0]}')
        else:
            return 'folder', fns

    data_type, aux = get_data_type(config.data_path)
    if data_type in ['mp4', 'npz']:
        dataset = get_dataset(
            config.data_path, data_type, batch_size, initial_shape=(N, batch_size // N)
        )
        class_map, mask_map = None, None
    else:
        data_paths = aux
        class_map = {osp.basename(k): i for i, k in enumerate(data_paths)}

        mask_file = osp.join(config.data_path, 'mask_map.pkl')
        if tf.io.gfile.exists(mask_file):
            mask_map = pickle.load(tf.io.gfile.GFile(mask_file, 'rb'))
        else:
            mask_map = None
        
        batch_per_dset = max(1, batch_size // len(data_paths))
        dataset_labels = list(range(len(data_paths)))
        if len(data_paths) >= num_ds_shards:
            data_paths = np.array_split(data_paths, num_ds_shards)[ds_shard_id].tolist()
            dataset_labels = np.array_split(dataset_labels, num_ds_shards)[ds_shard_id].tolist()

            # No need to shard further in load_* functions
            num_ds_shards = 1
            ds_shard_id = 0

        datasets = []
        for data_path in data_paths:
            data_type, _ = get_data_type(data_path)
            if mask_map is None:
                mask = None
            else:
                mask = mask_map[osp.basename(data_path)]
            dataset = get_dataset(data_path, data_type, batch_per_dset, initial_shape=None, mask=mask)
            datasets.append(dataset)

        def combine(datasets, dataset_labels):
            def _fn(*xs):
                x = np.concatenate(xs, axis=0)[:batch_size]
                x = x.reshape(N, batch_size // N, *x.shape[1:])
                return x

            idx = 0
            while True:
                batch = []
                for _ in range(math.ceil(batch_size / batch_per_dset)):
                    while datasets[idx] is None:
                        idx = (idx + 1) % len(datasets)
                    x_i = next(datasets[idx])
                    x_i['label'] = np.full(batch_per_dset, dataset_labels[idx], dtype=np.int32)
                    batch.append(x_i)
                    idx = (idx + 1) % len(datasets)
                batch = jax.tree_map(_fn, *batch)
                yield batch
        dataset = combine(datasets, dataset_labels)
        
    dataset = jax_utils.prefetch_to_device(dataset, 2)
    return dataset, class_map, mask_map


def prepare(dataset, batch_size, initial_shape=None):
    dataset = dataset.shuffle(batch_size if os.environ.get('DEBUG') == '1' else batch_size * 64)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    def prepare_tf_data(xs):
        def _prepare(x):
            x = x._numpy()
            if initial_shape is not None:
                x = x.reshape(*initial_shape, *x.shape[1:])
            return x
        xs = jax.tree_map(_prepare, xs)
        return xs
    iterator = map(prepare_tf_data, dataset)
    return iterator

 
def load_npz(config, data_path, train, n_shards, shard_id, mask):
    split = 'train' if train else 'test'
    folder = osp.join(data_path, split, '*.npz')
    fns = _glob_files(folder)
    random.Random(1234).shuffle(fns) 
    assert len(fns) > 0, f"Could not find any files for {folder}"
    fns = np.array_split(fns, n_shards)[shard_id].tolist()
    
    if mask is not None:
        mask = mask.astype(np.uint8)

    def read(path):
        path = path.decode('utf-8')
        if path.startswith('gs://'):
            path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        video = np.load(path)['arr_0']
        if mask is not None:
            video *= mask
        return video
    
    def process(video):
        if hasattr(config, 'seq_len'):
            req_len = 1 + (config.seq_len - 1) * config.frame_skip
            max_idx = tf.shape(video)[0] - req_len + 1
            max_idx = tf.minimum(max_idx, req_len)
            idx = tf.random.uniform((), 0, max_idx, dtype=tf.int32)
            video = video[idx:]

            video = video[:tf.shape(video)[0] // req_len * req_len]
            video = tf.reshape(
                video,
                tf.concat(([tf.shape(video)[0] // req_len, req_len], tf.shape(video)[1:]), 0)
            )
            video = video[:, ::config.frame_skip]
        else:
            video = video[None]
        video = tf.cast(video, tf.float32) / 127.5 - 1
        return tf.data.Dataset.from_tensor_slices(video)
    
    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.uint8]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.flat_map(process)
    dataset = dataset.map(lambda video: dict(video=video))
    return dataset


def load_mp4s(config, data_path, train, n_shards, shard_id, mask):
    split = 'train' if train else 'test'
    folder = osp.join(data_path, split, '*.mp4')
    fns = _glob_files(folder)
    random.Random(1234).shuffle(fns) 
    assert len(fns) > 0, f"Could not find any files for {folder}"
    fns = np.array_split(fns, n_shards)[shard_id].tolist()

    if mask is not None:
        mask = mask.astype(np.uint8)

    def read(path):
        data = tf.io.gfile.GFile(path, mode='rb').read()
        video = tfio.experimental.ffmpeg.decode_video(data)
        if mask is not None:
            video *= mask
        return video

    def process(video):
        if hasattr(config, 'seq_len'):
            req_len = 1 + (config.seq_len - 1) * config.frame_skip
            max_idx = tf.shape(video)[0] - req_len + 1
            max_idx = tf.minimum(max_idx, req_len)
            idx = tf.random.uniform((), 0, max_idx, dtype=tf.int32)
            video = video[idx:]

            video = video[:tf.shape(video)[0] // req_len * req_len]
            video = tf.reshape(
                video,
                tf.concat(([tf.shape(video)[0] // req_len, req_len], tf.shape(video)[1:]), 0)
            )
            video = video[:, ::config.frame_skip]
        else:
            video = video[None]
        video = tf.cast(video, tf.float32) / 127.5 - 1
        return video
    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(read, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(process)
    dataset = dataset.map(lambda video: dict(video=video))
    return dataset

    
def _glob_files(pattern):
    if pattern.startswith('gs://'):
        fns = tf.io.gfile.glob(pattern)
    else:
        fns = list(glob.glob(pattern))
    fns.sort()
    return fns
