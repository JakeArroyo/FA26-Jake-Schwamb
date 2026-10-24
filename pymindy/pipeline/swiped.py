import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
from pymindy.config import pyMINDyConfig
from pymindy.dataset import PartitionDataset
from pymindy.dataset.SCHDataset import SCHDatasetMeta
from pymindy.io import save_model
from pymindy.training.mindy_core import pyMINDyCoreMini
from pymindy.training.utils import train_data_MSE

torch._dynamo.config.capture_dynamic_output_shape_ops = True
# noinspection PyProtectedMember
torch._dynamo.config.capture_func_transforms = True
patients_to_process=["S004 TX2"]
obs_size = 20

input_version="v0.1"
output_version="v0.1"
config=pyMINDyConfig()
input_dir=os.path.join(config.get_input_folder_path("SWIPED-RAW"),"middle",input_version)
dataset_dir=config.get_output_folder_path("SWIPED-RAW")
output_dir = os.path.join(dataset_dir,"output",output_version)
param_range = {"Se_min": 0.5, "Se_max": 4, "Si_min": 0.5, "Si_max": 4, "De_max": 1, "De_min": 0, "Di_max": 1,
               "Di_min": 0}
# HPC
torch_device = "cuda"
cuda_num =1
parallel_process_num=32
parallel = True

### Data loading
mini_batch_size =128
shuffle = True
drop_last = True
### Optimizer
lr = 0.01
betas = (0.9, 0.999)
weight_decay = 2*1e-2
amsgrad = True
clipping_grad_norm = True
clipping_grad_norm_value = 4

# training parameters
n_batch = 100
stack_interval = 20
rand_scale = 1e-4  # initial scale of the parameters

# KF parameters
warm_up_step = 5
kf_step = 25
rec_step = 5
P0_scale = 5
Q_scale = 0.2
R_scale = 0.2



hyperparameters = {
    "obs_size": obs_size,
    "input_dir": input_dir,
    "output_dir": output_dir,
    "param_range": param_range,
    "torch_device": torch_device,
    "cuda_num": cuda_num,
    "parallel_process_num": parallel_process_num,
    "parallel": parallel,
    "mini_batch_size": mini_batch_size,
    "shuffle": shuffle,
    "drop_last": drop_last,
    "lr": lr,
    "betas": betas,
    "weight_decay": weight_decay,
    "amsgrad": amsgrad,
    "clipping_grad_norm": clipping_grad_norm,
    "clipping_grad_norm_value": clipping_grad_norm_value,
    "n_batch": n_batch,
    "stack_interval": stack_interval,
    "rand_scale": rand_scale,
    "warm_up_step": warm_up_step,
    "kf_step": kf_step,
    "rec_step": rec_step,
    "P0_scale": P0_scale,
    "Q_scale": Q_scale,
    "R_scale": R_scale
}


def prepare_data_loader(dataset, device):
    if device[:4] == "cuda":
        if parallel:
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=shuffle,
                pin_memory=True,
                pin_memory_device=device,
                drop_last=drop_last
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=shuffle,
                pin_memory=True,
                pin_memory_device=device,
                drop_last=drop_last,
                num_workers=10,
                persistent_workers=True
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=mini_batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
    return train_loader


if torch_device[:4]  == "cuda":
    torch.cuda.empty_cache()

n_pair = obs_size
H = np.hstack([np.eye(n_pair), np.zeros((n_pair, n_pair))])

input_meta = SCHDatasetMeta(input_dir)
os.makedirs(output_dir, exist_ok=True)


# noinspection PyShadowingNames
def process_epoch(x):
    start_time = datetime.now()
    i, input_path = x
    if torch_device == "cuda" and parallel:
        process_device = f"cuda:{i % cuda_num}"
    else:
        process_device = torch_device

    Q = Q_scale * np.eye(H.shape[1])
    R = R_scale * np.eye(H.shape[0])
    P0 = P0_scale * np.eye(
        H.shape[1])

    partition = PartitionDataset.load(input_path)
    data = partition.get_normalized_data()
    labels = partition.channel_labels.tolist()
    record_name = str(partition.record_name)
    subject_name = str(partition.subject_name)
    seg_num = int(partition.seg_num)

    output_folder = os.path.join(output_dir, str(record_name))
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{record_name}_par_{seg_num}.pt")
    if os.path.exists(output_path):
        print(f"Skipping {output_path}")
        return
    mindy_core = pyMINDyCoreMini.init(H, Q, R, device=process_device, rand_scale=rand_scale,
                                      param_range=param_range, fix_Se=False, fix_Ve=False, fix_Si=True, fix_Vi=True)
    optimizer = torch.optim.AdamW(mindy_core.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, fused=True,
                                  amsgrad=amsgrad)

    mindy_core, info = train_data_MSE(data,
                                      mindy_core, n_batch,
                                      stack_interval,
                                      warm_up_step,
                                      kf_step, rec_step,
                                      optimizer, P0,
                                      clipping_grad_norm=clipping_grad_norm,
                                      clipping_grad_norm_value=clipping_grad_norm_value,
                                      device=process_device,
                                      info=record_name + " " + str(seg_num) + " " + process_device,
                                      prepare_data_loader_func=prepare_data_loader)

    end_time = datetime.now()
    training_time_minutes = (end_time - start_time).total_seconds() / 60
    print(f"{record_name}-{seg_num}\tFinished:\t{end_time.strftime('%H:%M:%S (%m/%d/%Y)')}")
    print(f"{record_name}-{seg_num}\tDuration:\t{round(training_time_minutes, 1)} minutes\n")
    info |= {
        "subject_name": subject_name,
        "record_name": record_name,
        "epoch_num": seg_num,
        "params": mindy_core.get_params(),
        "input_path": input_path,
        "output_path": output_path,
        "label": labels,
        "training time": training_time_minutes,
        "process device": process_device,
        "hyperparameters": hyperparameters,
        "partition_meta": partition.meta
    }

    save_model(output_path, mindy_core, info)


if __name__ == '__main__':
    global_start_time = datetime.now()
    print(f"Program starts at {global_start_time.strftime('%H:%M:%S (%m/%d/%Y)')}\n\n")
    os.makedirs(output_dir, exist_ok=True)
    for patient in patients_to_process:
        subject = f"{patient}"

        subject_path=os.path.join(input_dir,subject)

        subject_records = sorted(os.listdir(subject_path), key=lambda x:(int(re.search(r"ep_(\d+).npz", x).group(1)), int(re.search(r"ep_(\d+).npz", x).group(1))))
        subject_record_path=[os.path.join(subject_path,record) for record in subject_records]
        print(f"Processing {subject} with {len(subject_record_path)} partitions...")
        if parallel:
            pool = mp.Pool(processes=parallel_process_num)
            results = pool.map(process_epoch, enumerate(subject_record_path),chunksize=1)
            pool.close()
            pool.join()
        else:
            for x in enumerate(subject_record_path):
                process_epoch(x)
    global_end_time = datetime.now()
    print(f"Program ends at {global_end_time.strftime('%H:%M:%S (%m/%d/%Y)')}")
    print("All done!")
