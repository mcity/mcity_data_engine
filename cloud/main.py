import datetime
import json
import os

from aws_stream_filter_framerate import SampleTimestamps
from s3_file_list import AwsDownloader


def main():
    # Prepare logging and storing
    storage_root = "."
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
        name= "mcity-data-engine",
        start_date=start_date,
        end_date=end_date,
        sample_rate_hz=sample_rate,
        test_run=True,
        storage_target_root=storage_root,
        subfolder_data=subfolder_data,
        subfolder_logs=subfolder_logs,
        log_time=log_time
    )
    file_names = aws_downloader.process_data()
    i = 0

if __name__ == "__main__":
    main()