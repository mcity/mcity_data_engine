import argparse
import datetime
import os
from datetime import datetime as dt

from s3_file_list import AwsDownloader


def valid_date(s):
    """Convert string to datetime object in YYYY-MM-DD format, raising ArgumentTypeError if invalid."""
    try:
        return dt.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = f"'{s}' is not a valid date in YYYY-MM-DD format"
        raise argparse.ArgumentTypeError(msg)


def main():
    """Main function to process AWS data with date range and sample rate parameters, handle data storage, and execute AWS download operations."""
    parser = argparse.ArgumentParser(
        description="Process AWS data with specific date range and sample rate"
    )
    parser.add_argument(
        "--start",
        type=valid_date,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end", type=valid_date, required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--rate", type=float, default=1.0, help="Sample rate in Hz (default: 1.0)"
    )

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


# Example call: python main.py --start 2023-11-19 --end 2023-11-25 --rate 1
# Multiple days can be processed in parallel by launching multiple instances of the script
if __name__ == "__main__":
    main()
