import logging
import os
import re
from datetime import datetime

import numpy as np
import torch

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_func_transforms = True

import torch.multiprocessing as mp
from pymindy import logger as pymindy_logger
from pymindy.config import pyMINDyConfig
from pymindy.dataset import PartitionDataset
from pymindy.dataset.sch_dataset import SCHDatasetMeta
from pymindy.io import save_model
from pymindy.training import pyMINDyCoreMini
from pymindy.training import train_data_MSE
from argparse import ArgumentParser

argument_parser = ArgumentParser(description="Train pyMINDy model on SCH dataset")
argument_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = argument_parser.parse_args()


config=pyMINDyConfig()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# logging level
if args.debug:
    logger.setLevel(logging.DEBUG)
    pymindy_logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
    pymindy_logger.setLevel(logging.INFO)

# Pipeline Configuration
patients_to_process = [5, 7, 17, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
torch_device = "cuda"
cuda_num = torch.cuda.device_count()  # auto detect
process_num = 30
skip_existing = False
max_try = 3

# Input dir
input_version = "v2.0"
input_dir=os.path.join(config.get_input_folder_path("SCH"),"middle",input_version)
input_meta = SCHDatasetMeta(input_dir)

# Output dir
output_version = "v2.0"
dataset_dir=config.get_output_folder_path("SCH")
output_dir = os.path.join(dataset_dir,"output",output_version)
os.makedirs(output_dir, exist_ok=True)

## Range of parameters
param_range = {"Se_min": 0.25, "Se_max": 4, "Si_min": 0.25, "Si_max": 4, "De_max": 1, "De_min": 0, "Di_max": 1,
               "Di_min": 0}

## Data configuration
stack_interval = 50
mini_batch_size = 256
n_batch = 50

### Optimizer configuration
lr = 0.05
betas = (0.9, 0.999)
weight_decay = 2 * 1e-2
amsgrad = True
clipping_grad_norm = True
clipping_grad_norm_value = 1

# Kalman Filter configuration
warm_up_step = 5
kf_step = 35
rec_step = 0
P0_scale = 5
Q_scale = 0.2
R_scale = 0.1

if args.debug:
    logger.info(f"Debug mode enabled.")
    cuda_num = 1
    logger.info("Overriding cuda_num to 1 for debugging.")
    process_num = 1
    logger.info("Overriding process_num to 1 for debugging.")

############################################

config_dict = {
    "patients_to_process": patients_to_process,
    "torch_device": torch_device,
    "cuda_num": torch.cuda.device_count(),

    "process_num": process_num,
    "skip_existing": skip_existing,
    "input_version": input_version,
    "input_dir": input_dir,
    "input_meta": input_meta,
    "output_version": output_version,
    "dataset_dir": dataset_dir,
    "output_dir": output_dir,
    "param_range": param_range,
    "stack_interval": stack_interval,
    "mini_batch_size": mini_batch_size,
    "n_batch": n_batch,
    "lr": lr,
    "betas": betas,
    "weight_decay": weight_decay,
    "amsgrad": amsgrad,
    "clipping_grad_norm": clipping_grad_norm,
    "clipping_grad_norm_value": clipping_grad_norm_value,
    "warm_up_step": warm_up_step,
    "kf_step": kf_step,
    "rec_step": rec_step,
    "P0_scale": P0_scale,
    "Q_scale": Q_scale,
    "R_scale": R_scale
}

def prepare_data_loader(dataset, device):
    if device[:4] == "cuda":
        if process_num >1:
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=mini_batch_size,
                    shuffle=True,
                pin_memory=True,
                pin_memory_device=device,
                    drop_last=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=mini_batch_size,
                    shuffle=True,
                pin_memory=True,
                pin_memory_device=device,
                    drop_last=True,
                num_workers=10,
                persistent_workers=True
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=mini_batch_size,
                shuffle=True,
                drop_last=True
        )
    return train_loader





def process_epoch(x):
    start_time = datetime.now()
    i, input_path = x
    if torch_device == "cuda" and process_num > 1:
        gpu_device = f"cuda:{i % cuda_num}"
    else:
        gpu_device = torch_device

    # Load meta
    partition = PartitionDataset.load(input_path)
    data = partition.get_normalized_data()
    labels = partition.channel_labels.tolist()

    subject_name = str(partition.subject_name)
    record_name = str(partition.record_name)
    # TODO Add this to future preprocessing step

    epoch_num = int(re.search(r".*_Epoch(\d+)", record_name).group(1))
    parition_num = int(partition.seg_num)
    logger.info(f"Processing subject {subject_name[7:]} epoch {epoch_num} partition {parition_num} on {gpu_device}")
    logger.debug(
        f"Meta data: subject_name: {subject_name}, record_name: {record_name}, epoch_num: {epoch_num}, "
        f"seg_num: {parition_num}")
    n_pair = data.shape[1]
    H = np.hstack([np.eye(n_pair), np.zeros((n_pair, n_pair))])
    Q = Q_scale * np.eye(H.shape[1])
    R = R_scale * np.eye(H.shape[0])
    P0 = P0_scale * np.eye(
            H.shape[1])

    output_path = os.path.join(output_dir, f"{record_name}_par_{parition_num}.pt")

    if skip_existing == True and os.path.exists(output_path):
        print(f"Skipping {output_path}")
        return
    mindy_core = pyMINDyCoreMini.init(
        H, Q, R, device=gpu_device,
        param_range=param_range, fix_Se=False, fix_Ve=False, fix_Si=True, fix_Vi=True, rand_scale=1e-4)
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
                                      device=gpu_device,
                                      info=f"pt{subject_name[7:]}-ep{epoch_num}-par{parition_num}-{gpu_device}",
                                      prepare_data_loader_func=prepare_data_loader)

    end_time = datetime.now()
    training_time_minutes = (end_time - start_time).total_seconds() / 60
    logger.info(f"pt{subject_name[7:]}-ep{epoch_num}-par{parition_num}-{gpu_device} Finished: {end_time.strftime('%H:%M:%S (%m/%d/%Y)')} Duration: {round(training_time_minutes, 1)} minutes\n")
    info |= {
        "subject_name": subject_name,
        "record_name": record_name,
        "partition_num": parition_num,
        "epoch_num": epoch_num,
        "partition_meta": partition.meta,
        "params": mindy_core.get_params(),
        "input_path": input_path,
        "output_path": output_path,
        "label": labels,
        "training time": training_time_minutes,
        "process device": gpu_device,
        "config": config_dict,

    }

    save_model(output_path, mindy_core, info)


def process_function(x):
    attempt = 0
    while attempt < max_try:
        try:
            process_epoch(x)
            return 1
        except Exception as e:
            logger.error(f"Error processing {x}: {e}")
            attempt += 1
            if attempt == max_try:
                logger.error(f"Max attempts reached for {x}. Skipping.")
                return 0



if __name__ == '__main__':
    start_time = datetime.now()
    logger.info(f"Program starts at {start_time.strftime('%H:%M:%S (%m/%d/%Y)')}\n\n")
    os.makedirs(output_dir, exist_ok=True)
    if torch_device == "cuda":
        torch.cuda.empty_cache()
    for patient in patients_to_process:
        subject = f"Patient{patient}"

        subject_path=os.path.join(input_dir,subject)

        subject_records = sorted(os.listdir(subject_path), key=lambda x:(int(re.search(r".*_Epoch(\d+)_par.*", x).group(1)), int(re.search(r".*par(\d+).npz", x).group(1))))
        subject_record_path=[os.path.join(subject_path,record) for record in subject_records]
        logger.info(f"Processing {subject} with {len(subject_record_path)} partitions...")
        if process_num > 1:
            pool = mp.Pool(processes=process_num)
            results = pool.map(process_function, enumerate(subject_record_path), chunksize=1)

            pool.close()
            pool.join()
        else:
            results = []
            for x in enumerate(subject_record_path):
                results.append(process_function(x))
    end_time = datetime.now()

    logger.info(f"Program ends at {end_time.strftime('%H:%M:%S (%m/%d/%Y)')}")
    total_tasks = len(results)
    successful_tasks = sum(results)
    failed_tasks = total_tasks - successful_tasks
    logger.info(f"Total tasks: {total_tasks}, Successful tasks: {successful_tasks}, Failed tasks: {failed_tasks}")
    logger.info("All done!")