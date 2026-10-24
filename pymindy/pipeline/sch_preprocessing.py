import os
import re
from argparse import ArgumentParser

import mne
import torch.multiprocessing as mp
from pandas import Timedelta
from pymindy.config import pyMINDyConfig
from pymindy.dataset import PartitionDataset
from pymindy.dataset.SCHDataset import SCHDatasetMeta

arg_parser = ArgumentParser()
arg_parser.add_argument("--process_number", type=int, default=32)

args = arg_parser.parse_args()
process_number = args.process_number   
mne.set_log_level("ERROR")

version = "v2.0"

config=pyMINDyConfig()

if config.get_env()=="local":
    input_dir="/Volumes/active/Eisenman_EEG_Data/Multimodal_2023"
else:
    input_dir="/storage1/fs1/shinung/Active/Eisenman_EEG_Data/Multimodal_2023"
dataset_meta=SCHDatasetMeta(input_dir)

dataset_root = config.get_output_folder_path("SCH")  # zongxi/dataset, with "RIS2" as default data source
output_dir=os.path.join(dataset_root,"middle",version)

subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  #
interval_minutes = 15


if __name__ == '__main__':
    for x in subjects:
        subject=f"Patient{x}"
        l_freq=0.5
        h_freq = 30
        channels = ["C3", "C4", "Cz", "F3", "F4", "F7", "F8", "F9", "F10", "Fp1", "Fp2", "Fpz", "Fz", "O1", "O2", "Oz",
                    "P3", "P4", "P7", "P8", "Pz", "T7", "T8"]
    
    
        print(f"Start processing {subject}:")
        subject_folder_path=dataset_meta.get_subject_folder(subject)
        subject_edf_files = sorted(list(filter(lambda y: y[-3:] == "edf", os.listdir(subject_folder_path))),
                                   key=lambda y: int(re.search(r".*_Epoch(\d+).edf", y).group(1)))
        subject_output_dir=os.path.join(output_dir,subject)
        os.makedirs(subject_output_dir,exist_ok=True)
        def process_edf_file(edf_file):
            print(f"Processing {edf_file}...")
            edf_file_path=os.path.join(subject_folder_path,edf_file)
            raw = mne.io.read_raw_edf(edf_file_path, preload=True,
                                      include=r"EEG\s([a-zA-Z0-9_.-]*)-Ref").rename_channels(
                lambda y: re.search(r"EEG\s([a-zA-Z0-9_.-]*)-Ref", y).group(1)).set_eeg_reference(
                ref_channels=["A1", "A2"]).pick(channels)
            n_times = raw.n_times
            fs = int(raw.info["sfreq"])
            onset_time = raw.info["meas_date"]
            filtered_raw = raw.filter(l_freq=l_freq, h_freq=h_freq).notch_filter(freqs=[60, 120], method="spectrum_fit")
            df=filtered_raw.to_data_frame()
            df.drop(columns=["time"],inplace=True)
            df=df[channels]
            mean=df.mean(axis=0).values
            std=df.std(axis=0).values
    
            start=list(range(0,n_times,fs*60*interval_minutes))
            end=start[1:]+[n_times]
    
            if start[-1]>=end[-1]:
                start=start[:-1]
                end=end[:-1]
            for i,(s,e) in enumerate(zip(start,end)):
                meta = {"raw_file_name": edf_file, "start_time": onset_time + Timedelta(seconds=s // fs),
                        "end_time": onset_time + Timedelta(seconds=(e - 1) // fs), "fs": fs}
                partition_df=df.iloc[s:e]
                partition=PartitionDataset(partition_df.values, channels, edf_file[:-4], i, subject, mean, std, meta)
                partition_name=edf_file[:-4]+"_par"+str(i)+".npz"
                partition_output_path=os.path.join(subject_output_dir,partition_name)
                partition.save(partition_output_path)
                print(f"\tSaved to {partition_output_path}!")


        if process_number is not None and process_number > 1:
            pool = mp.Pool(processes=process_number)
            results = pool.map(process_edf_file, subject_edf_files)
            pool.close()
            pool.join()
        else:
            for file in subject_edf_files:
                process_edf_file(file)

    print("Finish!")

