import base64
import datetime
import json
import os
import re
import shutil
import time
import traceback

import boto3
import wandb
from aws_stream_filter_framerate import SampleTimestamps
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class AwsDownloader:

    def __init__(
        self,
        name: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        sample_rate_hz: float,
        log_time: datetime.datetime,
        source: str = "mcity_gridsmart",
        storage_target_root: str = ".",
        subfolder_data: str = "data",
        subfolder_logs: str = "logs",
        test_run: bool = False,
        delete_old_data: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.sample_rate_hz = sample_rate_hz
        self.source = source
        self.storage_target_root = storage_target_root
        self.test_run = test_run
        self.delete_old_data = delete_old_data
        self.log_time = log_time
        self.file_names = []

        self.log_download = {}
        self.log_sampling = {}

        # Fill log
        self.log_download["source"] = source
        self.log_download["sample_rate_hz"] = sample_rate_hz
        self.log_download["storage_target_root"] = storage_target_root
        self.log_download["selection_start_date"] = start_date.strftime("%Y-%m-%d")
        self.log_download["selection_end_date"] = end_date.strftime("%Y-%m-%d")
        self.log_download["delete_old_data"] = delete_old_data
        self.log_download["test_run"] = test_run

        # Run name
        formatted_start = self.start_date.strftime("%Y-%m-%d")
        formatted_end = self.end_date.strftime("%Y-%m-%d")
        self.run_name = f"data_engine_rolling_{formatted_start}_to_{formatted_end}"

        # Setup storage folders
        self.data_target = os.path.join(storage_target_root, subfolder_data, self.run_name)
        os.makedirs(self.data_target, exist_ok=True)

        self.log_target = os.path.join(storage_target_root, subfolder_logs, self.run_name)
        os.makedirs(self.log_target, exist_ok=True)

        # Connect to AWS S3
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", None),
            region_name="us-east-1",
        )

    def process_data(self):
        cameras_dict = self._mcity_init_cameras()
        self._mcity_process_aws_buckets(cameras_dict)
        self.file_names, n_files_to_download = self._mcity_select_data(cameras_dict)

        # Tracking
        wandb.init(
            name=self.run_name,
            job_type="download",
            project="Data Engine Download",
        )

        targets = []
        step = 0
        with tqdm(desc="Processing data", total=n_files_to_download) as pbar:
            for camera in cameras_dict:
                for aws_source in cameras_dict[camera]["aws-sources"]:
                    bucket = aws_source.split("/", 1)[0]
                    for date in cameras_dict[camera]["aws-sources"][aws_source]:
                        for file in cameras_dict[camera]["aws-sources"][aws_source][date]:
                            try:
                                log_run = {}

                                # AWS S3 Download
                                time_start = time.time()
                                file_name = os.path.basename(file)
                                key = cameras_dict[camera]["aws-sources"][aws_source][
                                    date
                                ][file_name]["key"]
                                target = "./data/" + file_name
                                targets.append(target)
                                self.s3.download_file(bucket, key, target)

                                # Logging
                                file_size_mb = cameras_dict[camera]["aws-sources"][
                                    aws_source
                                ][date][file_name]["size"] / (1024**2)
                                time_end = time.time()
                                duration = time_end - time_start
                                mb_per_s = file_size_mb / duration
                                wandb.log({"download/mb_per_s": mb_per_s}, step)
                                wandb.log({"download/s": duration}, step)

                                # Sample data
                                time_start = time.time()
                                sampler = SampleTimestamps(
                                    file_path=target, target_framerate_hz=1
                                )
                                timestamps = sampler.get_timestamps()

                                # We need at least 2 timestamps to calculate a framerate
                                if (len(timestamps) >= 2):  
                                    # Get framerate
                                    framerate_hz, timestamps, upper_bound_threshold = (
                                        sampler.get_framerate(timestamps, log_run)
                                    )
                                    valid_target_framerate = (
                                        sampler.check_target_framerate(
                                            framerate_hz, log_run
                                        )
                                    )
                                    # We need a target framerate lower than the oririinal framerate
                                    if valid_target_framerate:
                                        # Sample data
                                        (
                                            selected_indices,
                                            selected_timestamps,
                                            target_timestamps,
                                            selected_target_timestamps,
                                        ) = sampler.sample_timestamps(
                                            timestamps, upper_bound_threshold, log_run
                                        )

                                        time_end = time.time()
                                        duration = time_end - time_start
                                        timestamps_per_s = len(timestamps) / duration
                                        wandb.log({"sampling/timestamps_per_s": timestamps_per_s}, step)
                                        wandb.log({"sampling/s": duration}, step)

                                        # Upload data
                                        time_start = time.time()
                                        file_size_mb = sampler.update_upload_file(
                                            target, selected_indices
                                        )

                                        time_end = time.time()
                                        duration = time_end - time_start
                                        mb_per_s = file_size_mb / duration
                                        wandb.log({"upload/mb_per_s": mb_per_s}, step)
                                        wandb.log({"upload/s": duration}, step)


                                    # Update log
                                    self.log_sampling[file] = log_run

                                else:
                                    print(
                                        f"Not enough timestamps to calculate framerate. Skipping {file}"
                                    )

                                # Delete local data
                                os.remove(target)
                                os.remove(target + "_sampled_1Hz")

                                # Update progress bar
                                step += 1
                                pbar.update(1)
                        
                            except Exception as e:
                                print(f"Error in mcity_gridsmart_loader: {e}")
                                print(traceback.format_exc())

        # Finish tracking
        wandb.finish()
        pbar.close()

        # Store download log
        name_log_download = "FileDownload"
        self.log_download["data"] = cameras_dict
        log_name = (self.log_time + "_" + name_log_download).replace(
            " ", "_"
        ).replace(":", "_") + ".json"
        log_file_path = os.path.join(self.log_target, log_name)
        with open(log_file_path, "w") as json_file:
            json.dump(self.log_download, json_file, indent=4)

        # Store sampling log
        name_log_sampling = "FileSampling"
        log_name = (self.log_time + "_" + name_log_sampling).replace(
            " ", "_"
        ).replace(":", "_") + ".json"
        log_file_path = os.path.join(self.log_target, log_name)
        with open(log_file_path, "w") as json_file:
            json.dump(self.log_sampling, json_file, indent=4)

    def _mcity_init_cameras(
        self,
        cameras={
            "Geddes_Huron_1",
            "Geddes_Huron_2",
            "Huron_Plymouth_1",
            "Huron_Plymouth_2",
            "Main_stadium_1",
            "Main_stadium_2",
            "Plymouth_Beal",
            "Plymouth_Bishop",
            "Plymouth_EPA",
            "Plymouth_Georgetown",
            "State_Ellsworth_NE",
            "State_Ellsworth_NW",
            "State_Ellsworth_SE",
            "State_Ellsworth_SW",
            "Fuller_Fuller_CT",
            "Fuller_Glazier_1",
            "Fuller_Glazier_2",
            "Fuller_Glen",
            "Dexter_Maple_1",
            "Dexter_Maple_2",
            "Hubbard_Huron_1",
            "Hubbard_Huron_2",
            "Maple_Miller_1",
            "Maple_Miller_2",
        },
    ):

        cameras_dict = {camera.lower(): {} for camera in cameras}
        for id, camera in enumerate(cameras_dict):
            cameras_dict[camera]["id"] = id
            cameras_dict[camera]["aws-sources"] = {}

        print(f"Processed {len(cameras_dict)} cameras")
        return cameras_dict

    def _mcity_process_aws_buckets(
        self,
        cameras_dict,
        aws_sources={
            "sip-sensor-data": [""],
            "sip-sensor-data2": ["wheeler1/", "wheeler2/"],
        },
    ):
        for bucket in tqdm(aws_sources, desc="Processing AWS sources"):
            for folder in aws_sources[bucket]:
                # Get and pre-process AWS data
                result = self.s3.list_objects_v2(
                    Bucket=bucket, Prefix=folder, Delimiter="/"
                )
                folders = self._process_aws_result(result)

                # Align folder names with camera names
                folders_aligned = [
                    re.sub(r"(?<!_)(\d)", r"_\1", folder.lower().rstrip("/"))
                    for folder in folders
                ]  # Align varying AWS folder names with camera names
                folders_aligned = [
                    folder.replace("fullerct", "fuller_ct")
                    for folder in folders_aligned
                ]  # Replace "fullerct" with "fuller_ct" to align with camera names
                folders_aligned = [
                    folder.replace("fullser", "fuller") for folder in folders_aligned
                ]  # Fix "fullser" typo in AWS

                # Check cameras for AWS sources
                if folders:
                    for camera_name in cameras_dict:
                        for folder_name, folder_name_aligned in zip(
                            folders, folders_aligned
                        ):
                            if (
                                camera_name in folder_name_aligned
                                and "gs_" in folder_name_aligned
                            ):  # gs_ is the prefix used in AWS
                                aws_source = f"{bucket}/{folder_name}"
                                if (
                                    aws_source
                                    not in cameras_dict[camera_name]["aws-sources"]
                                ):
                                    cameras_dict[camera_name]["aws-sources"][
                                        aws_source
                                    ] = {}
                else:
                    print(f"AWS did not return a list of folders for {bucket}/{folder}")

    def _mcity_select_data(self, cameras_dict):
        n_cameras = 0
        n_aws_sources = 0
        n_files_to_download = 0
        download_size_bytes = 0
        for camera in tqdm(cameras_dict, desc="Looking for data entries in range"):
            n_cameras += 1
            for aws_source in cameras_dict[camera]["aws-sources"]:
                file_downloaded_test = False
                n_aws_sources += 1
                bucket = aws_source.split("/")[0]
                prefix_camera = "/".join(aws_source.split("/")[1:])
                result = self.s3.list_objects_v2(
                    Bucket=bucket, Prefix=prefix_camera, Delimiter="/"
                )
                # Each folder represents a day
                folders_day = self._process_aws_result(result)
                for folder_day in folders_day:
                    date = folder_day.split("/")[-2]
                    if self.test_run:
                        # Choose a sample irrespective of the data range to get data from all AWS sources
                        in_range = True
                    else:
                        # Only collect data within the date range
                        timestamp = datetime.datetime.strptime(date, "%Y-%m-%d")
                        in_range = self.start_date <= timestamp <= self.end_date

                    if in_range:
                        cameras_dict[camera]["aws-sources"][aws_source][date] = {}
                        result = self.s3.list_objects_v2(
                            Bucket=bucket, Prefix=folder_day, Delimiter="/"
                        )
                        # Each folder represents an hour
                        folders_hour = self._process_aws_result(result)
                        for folder_hour in folders_hour:
                            result = self.s3.list_objects_v2(
                                Bucket=bucket, Prefix=folder_hour, Delimiter="/"
                            )
                            files = result["Contents"]
                            for file in files:
                                n_files_to_download += 1
                                download_size_bytes += file["Size"]
                                file_name = os.path.basename(file["Key"])
                                cameras_dict[camera]["aws-sources"][aws_source][date][
                                    file_name
                                ] = {}
                                cameras_dict[camera]["aws-sources"][aws_source][date][
                                    file_name
                                ]["key"] = file["Key"]
                                self.file_names.append(file["Key"])
                                cameras_dict[camera]["aws-sources"][aws_source][date][
                                    file_name
                                ]["size"] = file["Size"]
                                cameras_dict[camera]["aws-sources"][aws_source][date][
                                    file_name
                                ]["date"] = file["LastModified"].strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                if self.test_run:
                                    file_downloaded_test = True
                                    print(f"{aws_source} : {file_name}")
                                    break  # escape for file in files
                            if self.test_run and file_downloaded_test:
                                break  # escape for folder_hour in folders_hour
                        if self.test_run and file_downloaded_test:
                            break  # escape for folder_day in folders_day

        self.log_download["n_cameras"] = n_cameras
        self.log_download["n_aws_sources"] = n_aws_sources
        self.log_download["n_files_to_process"] = n_files_to_download
        self.log_download["selection_size_tb"] = download_size_bytes / (1024**4)

        return self.file_names, n_files_to_download

    def _mcity_download_data(self, cameras_dict, n_files_to_download, passed_checks):
        mb_per_s_list = []

        if passed_checks:
            download_successful = True

            # if self.delete_old_data:
            #     try:
            #         shutil.rmtree(self.data_target)
            #     except:
            #         pass
            #     os.makedirs(self.data_target)

            step = 0
            download_started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_download["download_started"] = download_started
            with tqdm(desc="Downloading data", total=n_files_to_download) as pbar:
                for camera in cameras_dict:
                    for aws_source in cameras_dict[camera]["aws-sources"]:
                        bucket = aws_source.split("/", 1)[0]
                        for date in cameras_dict[camera]["aws-sources"][aws_source]:
                            for file in cameras_dict[camera]["aws-sources"][aws_source][
                                date
                            ]:
                                time_start = time.time()

                                # AWS S3 Download
                                file_name = os.path.basename(file)
                                key = cameras_dict[camera]["aws-sources"][aws_source][
                                    date
                                ][file_name]["key"]
                                target = os.path.join(self.data_target, file_name)
                                self.s3.download_file(bucket, key, target)

            download_ended = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_download["download_ended"] = download_ended

        else:
            download_successful = False
            print("Safety checks failed. Not downloading data")

        return download_successful

    def _process_aws_result(self, result):
        # Get list of folders from AWS response
        if "CommonPrefixes" in result:
            folders = [prefix["Prefix"] for prefix in result["CommonPrefixes"]]
            return folders
        else:
            return None
