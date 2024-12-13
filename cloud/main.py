import argparse
import datetime
import json
import os
from datetime import datetime as dt

from aws_stream_filter_framerate import SampleTimestamps
from s3_file_list import AwsDownloader


def valid_date(s):
    try:
        return dt.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = f"'{s}' is not a valid date in YYYY-MM-DD format"
        raise argparse.ArgumentTypeError(msg)


def main():
    parser = argparse.ArgumentParser(description='Process AWS data with specific date range and sample rate')
    parser.add_argument('--start', type=valid_date, required=True,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=valid_date, required=True,
                      help='End date in YYYY-MM-DD format')
    parser.add_argument('--rate', type=float, default=1.0,
                      help='Sample rate in Hz (default: 1.0)')
    
    args = parser.parse_args()

    # Prepare logging and storing
    storage_root = "."
    subfolder_data = "data"
    subfolder_logs = "logs"
    storage_logs = os.path.join(storage_root, subfolder_logs)
    storage_data = os.path.join(storage_root, subfolder_data)
    os.makedirs(storage_logs, exist_ok=True)
    os.makedirs(storage_data, exist_ok=True)

    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    aws_downloader = AwsDownloader(
        name="mcity-data-engine",
        start_date=args.start,
        end_date=args.end,
        sample_rate_hz=args.rate,
        test_run=False,
        storage_target_root=storage_root,
        subfolder_data=subfolder_data,
        subfolder_logs=subfolder_logs,
        log_time=log_time,
    )

    # Load data, sample data, upload data, delete data
    aws_downloader.process_data()

if __name__ == "__main__":
    main()