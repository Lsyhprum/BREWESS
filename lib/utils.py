import os
import glob
import torch
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1,
                        allow_incomplete=True, callback=lambda x:x):
    indices = np.arange(len(tensors[0]))
    upper_bound = len(indices) // batch_size * batch_size + allow_incomplete * batch_size - 1
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break

def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out

def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x

def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    import gc, time
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)

def get_latest_file(pattern):
    list_of_files = glob.glob(pattern) # * means all if need specific format then *.csv
    assert len(list_of_files) > 0, "No files found: " + pattern
    return max(list_of_files, key=os.path.getctime)