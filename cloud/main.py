import datetime
import json
import os

from aws_stream_filter_framerate import SampleTimestamps
from s3_file_list import AwsDownloader


def main():
    print('Welcome')

    # Prepare logging and storing
    storage_root: str = "."    
    subfolder_data = "data"
    subfolder_logs = "logs"                     
    storage_logs = os.path.join(storage_root, subfolder_logs) 
    storage_data = os.path.join(storage_root, subfolder_data) 
    os.makedirs(storage_logs, exist_ok=True)  
    os.makedirs(storage_data, exist_ok=True) 

    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    start_date= datetime.datetime.strptime("2023-11-19", "%Y-%m-%d")
    end_date= datetime.datetime.strptime("2023-11-19", "%Y-%m-%d")
    sample_rate= 1
    aws_downloader = AwsDownloader(
                name= 'FileSelection',
                start_date=start_date,
                end_date=end_date,
                sample_rate_hz=sample_rate,
                test_run=True,
                storage_root=storage_root,
                subfolder_data=subfolder_data,
                subfolder_logs=subfolder_logs,
                log_time=log_time
            )
    file_names = aws_downloader.get_list()
    i = 0
    
    # Logging of sampling
    log_sampling = {}

    for file_name in file_names:
        log_run = {}
        sampler = SampleTimestamps(file_path=file_name, target_framerate_hz=1)
        framerate_hz, timestamps, upper_bound_threshold = sampler.get_framerate(log_run)
        valid_target_framerate = sampler.check_target_framerate(framerate_hz, log_run)
        if valid_target_framerate:
            selected_indices, selected_timestamps, target_timestamps, selected_target_timestamps = sampler.sample_timestamps(timestamps, upper_bound_threshold, log_run)
            sampler.update_upload_file(file_name,selected_indices)
            log_sampling[file_name] = log_run

    # Store log
    name_sampling = "FileSampling"
    log_name = (log_time + "_" + name_sampling).replace(" ", "_").replace(
        ":", "_"
    ) + ".json"
    log_file_path = os.path.join(storage_logs, log_name)
    with open(log_file_path, "w") as json_file:
        json.dump(log_sampling, json_file, indent=4)

if __name__ == "__main__":
    main()