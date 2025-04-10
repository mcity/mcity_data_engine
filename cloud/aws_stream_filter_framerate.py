import datetime
import json
import os
import shutil

import boto3
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "mcity-data-engine")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
region_name = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
)
s3 = session.client("s3")


class SampleTimestamps:

    def __init__(
        self, file_path: str, target_framerate_hz=1, aws_bucket=None, aws_prefix=None
    ):
        self.file_path = file_path
        self.aws_bucket = aws_bucket
        self.aws_prefix = aws_prefix
        self.target_framerate_hz = target_framerate_hz
        self.valid_target_framerate = True
        self.current_framerate_hz = None
        self.timestamps = []

        if file_path:
            self.execution_mode = "local"
        elif aws_bucket and aws_prefix:
            self.execution_mode = "aws"
        else:
            raise ValueError(
                "Either file_path or aws_bucket and aws_prefix must be provided"
            )

    def get_timestamps(self):
        timestamps = []

        with open(self.file_path, "r") as file:
            total_lines = sum(1 for _ in file)

        with open(self.file_path, "r") as file:
            # Collecting data
            for index, line in enumerate(file):
                data = json.loads(line)
                if "time" in data and "data" in data:
                    # Get time data
                    timestamp_raw = data.get("time")
                    try:
                        # Most timestamps have milliseconds
                        timestamp = datetime.datetime.strptime(
                            timestamp_raw, "%Y-%m-%d %H:%M:%S.%f"
                        )
                    except:
                        # Timestamps without milliseconds
                        timestamp = datetime.datetime.strptime(
                            timestamp_raw, "%Y-%m-%d %H:%M:%S"
                        )

                elif (
                    "image" in data
                    and "sensor_name" in data
                    and "event_timestamp" in data
                ):
                    # Get time data
                    timestamp_raw = data.get("event_timestamp")
                    timestamp = datetime.datetime.fromtimestamp(
                        timestamp_raw, tz=datetime.timezone.utc
                    )

                timestamps.append((index, timestamp))

        return timestamps

    def get_framerate(self, timestamps, log):

        # Calculate time differences (s) and current framerate (Hz)
        time_differences = []
        timestamps = sorted(timestamps, key=lambda x: x[1])

        previous_time = None
        for index, timestamp in timestamps:
            if previous_time is not None:
                time_difference = (timestamp - previous_time).total_seconds()
                time_differences.append(time_difference)
            previous_time = timestamp

        # Statistics about timestamp distribution
        average_time_diff = np.mean(time_differences)
        median_time_diff = np.median(time_differences)
        std_time_diff = np.std(time_differences)
        min_time_diff = np.min(time_differences)
        max_time_diff = np.max(time_differences)
        range_time_diff = max_time_diff - min_time_diff
        q1_time_diff = np.percentile(time_differences, 25)
        q3_time_diff = np.percentile(time_differences, 75)
        current_framerate_hz = 1 / median_time_diff  # Median is more robust to outliers

        log["time_s_avg_between_timestamps"] = average_time_diff
        log["time_s_median_between_timestamps"] = median_time_diff
        log["time_s_std_between_timestamps"] = std_time_diff
        log["time_s_min_between_timestamps"] = min_time_diff
        log["time_s_max_between_timestamps"] = max_time_diff
        log["time_s_range_between_timestamps"] = range_time_diff
        log["time_s_25_percentile_between_timestamps"] = q1_time_diff
        log["time_s_75_percentile_between_timestamps"] = q3_time_diff
        log["framerate_hz_original"] = current_framerate_hz
        log["framerate_hz_target"] = self.target_framerate_hz

        # Compute threshold
        interquartile_range = q3_time_diff - q1_time_diff  # Interquartile Range
        upper_bound_threshold = (
            q3_time_diff + 1.5 * interquartile_range
        )  # (1.5 * IQR rule)
        log["upper_bound_threshold"] = upper_bound_threshold

        return current_framerate_hz, timestamps, upper_bound_threshold

    def check_target_framerate(self, current_framerate_hz, log):
        # Check if target framerate is valid
        if self.target_framerate_hz > current_framerate_hz:
            print(
                f"Target framerate of {self.target_framerate_hz} Hz cannot exceed original framerate of {current_framerate_hz} Hz"
            )
            log["framerate_target_ok"] = False
            return False
        else:
            log["framerate_target_ok"] = True
            return True

    def sample_timestamps(self, timestamps, threshold_to_target, log):
        # Generate target timestamps
        start_time = timestamps[0][1]
        end_time = timestamps[-1][1]
        target_timestamps = []
        current_time = start_time
        while current_time <= end_time:
            target_timestamps.append(current_time)
            current_time += datetime.timedelta(seconds=1 / self.target_framerate_hz)

        # Find nearest original timestamps to target_timestamps_seconds
        selected_indices = []
        selected_timestamps = []
        selected_target_timestamps = []

        for target in target_timestamps:
            # Compute the time difference with each original timestamp
            time_diffs = [(target - t).total_seconds() for i, t in timestamps]
            time_diffs = np.abs(time_diffs)  # Take absolute differences

            # Find the index of the nearest timestamp
            nearest_index = np.argmin(time_diffs)

            # Ensure no duplicates are selected
            if (
                time_diffs[nearest_index] <= threshold_to_target
                and nearest_index not in selected_indices
            ):
                selected_target_timestamps.append(target)
                selected_indices.append(nearest_index)
                selected_timestamps.append(timestamps[nearest_index])

        # Log statistics
        log["n_original_timestamps"] = len(timestamps)
        log["n_target_timestamps"] = len(target_timestamps)
        log["n_selected_timestamps"] = len(selected_timestamps)

        if len(selected_timestamps) >=2:
            # Compute new framerate
            time_differences_new = []
            timestamps_new = sorted(selected_timestamps, key=lambda x: x[1])

            previous_time = None
            for index, timestamp in timestamps_new:
                if previous_time is not None:
                    time_difference = (timestamp - previous_time).total_seconds()
                    time_differences_new.append(time_difference)
                previous_time = timestamp
            median_time_diff_new = np.median(time_differences_new)
            new_framerate_hz = 1 / median_time_diff_new
            log["framerate_hz_sampled"] = new_framerate_hz

        else:
            print(f"Not enough selected timestamps ({len(selected_timestamps)}) to compute new framerate. Original number of timestamps: {len(timestamps)}")

        return (
            selected_indices,
            selected_timestamps,
            target_timestamps,
            selected_target_timestamps,
        )

    def update_upload_file(self, file_name, selected_indices, aws_source_path):
        output_file_path = file_name + f"_sampled_{self.target_framerate_hz}Hz"
        lines = []
        with open(file_name, "r") as file:
            for index, line in enumerate(file):
                if index in selected_indices:
                    lines.append(line)

        with open(output_file_path, "w") as output_file:
            output_file.writelines(lines)

        try:
            head, tail = os.path.split(output_file.name)
            file_size_mb = os.path.getsize(output_file.name) / (1024 * 1024)
            s3.upload_file(
                output_file.name,
                S3_BUCKET_NAME,
                str(self.target_framerate_hz) + "/" + aws_source_path +"_"+ tail,
            )
            return file_size_mb

        except Exception as e:
            print(
                "S3 upload failed for file "
                + str(str(self.target_framerate_hz) + "/" + tail)
                + " - "
                + str(e)
            )
            return None
