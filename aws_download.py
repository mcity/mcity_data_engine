import base64
import datetime
import json
import logging
import os
import re
import shutil
import time

import boto3
import fiftyone as fo
import pkg_resources
import pytz
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import NUM_WORKERS


class AwsDownloader:

    def __init__(
        self,
        name: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        sample_rate_hz: float,
        source: str = "mcity_gridsmart",
        storage_target_root: str = "/media/dbogdoll/Datasets/data_engine_rolling",
        test_run: bool = False,
        delete_old_data: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.sample_rate_hz = sample_rate_hz
        self.source = source
        self.storage_target_root = storage_target_root
        self.test_run = test_run
        self.delete_old_data = delete_old_data
        self.name = name

        self.log = {}

        # Fill log
        self.log["source"] = source
        self.log["sample_rate_hz"] = sample_rate_hz
        self.log["storage_target_root"] = storage_target_root
        self.log["selection_start_date"] = start_date.strftime("%Y-%m-%d")
        self.log["selection_end_date"] = end_date.strftime("%Y-%m-%d")
        self.log["delete_old_data"] = delete_old_data
        self.log["test_run"] = test_run

        # Setup storage folders
        self.data_target = os.path.join(storage_target_root, "data")
        os.makedirs(self.data_target, exist_ok=True)

        self.log_target = os.path.join(storage_target_root, "logs")
        os.makedirs(self.log_target, exist_ok=True)

        # Connect to AWS
        # Make sure you have the AWS credentials in your .secret file
        with open(".secret", "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                os.environ[key] = value

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", None),
        )

    def load_data(self):
        if self.source == "mcity_gridsmart":
            v51_dataset = self._mcity_gridsmart_loader()
            return v51_dataset
        else:
            logging.error(f"{self.loader} is not supported.")

    def _mcity_gridsmart_loader(self):
        try:
            cameras_dict = self._mcity_init_cameras()
            self._mcity_process_aws_buckets(cameras_dict)
            n_files_to_download, download_size_bytes = self._mcity_select_data(
                cameras_dict
            )
            passed_checks = self._mcity_safety_checks(cameras_dict, download_size_bytes)
            download_successful = self._mcity_download_data(
                cameras_dict, n_files_to_download, passed_checks
            )
            download_check_successful = self._mcity_check_downloaded_data(cameras_dict)
            if download_successful and download_check_successful:
                self._mcity_set_v51_metadata()
                self._mcity_unpack_data(cameras_dict)
                v51_dataset = self._mcity_create_v51_dataset(delete_old_dataset=True)
                return v51_dataset
            else:
                logging.error("Download failed. Check logs for details.")
        except Exception as e:
            logging.error(f"Error in mcity_gridsmart_loader: {e}")
            logging.error("Stack trace:", exc_info=True)
            return
        finally:
            self.log["data"] = cameras_dict
            log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_name = (log_time + "_" + self.name).replace(" ", "_").replace(
                ":", "_"
            ) + ".json"
            log_file_path = os.path.join(self.log_target, log_name)
            with open(log_file_path, "w") as json_file:
                json.dump(self.log, json_file, indent=4)

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

        logging.info(f"Processed {len(cameras_dict)} cameras")
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
                    logging.warning(
                        f"AWS did not return a list of folders for {bucket}/{folder}"
                    )

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
                                    logging.info(f"{aws_source} : {file_name}")
                                    break  # escape for file in files
                            if self.test_run and file_downloaded_test:
                                break  # escape for folder_hour in folders_hour
                        if self.test_run and file_downloaded_test:
                            break  # escape for folder_day in folders_day

        logging.info(f"Found {n_cameras} cameras")
        logging.info(f"Found {n_aws_sources} AWS sources")
        logging.info(f"Found {n_files_to_download} files to download")
        self.log["n_files_to_download"] = n_files_to_download
        self.log["download_size_tb"] = download_size_bytes / (1024**4)
        return n_files_to_download, download_size_bytes

    def _mcity_safety_checks(
        self, cameras_dict, download_size_bytes, MAX_SIZE_TB=1.5, MAX_DAYS=7
    ):
        # Safety checks prior to downloading
        passed_checks = True

        # Check if each camera was assigned at least one AWS source
        for camera in cameras_dict:
            if len(cameras_dict[camera]["aws-sources"]) == 0:
                logging.error(f"Camera {camera} has no AWS sources")
                passed_checks = False

        # Check if the total size of the data to download is within the limit
        total_size_tb = download_size_bytes / (1024**4)
        if total_size_tb > MAX_SIZE_TB:
            logging.error(
                f"Total size {total_size_tb} TB exceeds {MAX_SIZE_TB} TB limit"
            )
            passed_checks = False
        else:
            logging.info(
                f"Total size {total_size_tb} TB is within {MAX_SIZE_TB} TB limit"
            )

        # Check if the date range is within the limit (# +1 to include the end date)
        days_delta = (self.end_date - self.start_date).days + 1
        if (days_delta) > MAX_DAYS:
            logging.error(
                f"Date range of {days_delta} days exceeds {MAX_DAYS} days limit"
            )
            passed_checks = False
        else:
            logging.info(
                f"Date range of {days_delta} days is within {MAX_DAYS} days limit"
            )
        return passed_checks

    def _mcity_download_data(self, cameras_dict, n_files_to_download, passed_checks):
        mb_per_s_list = []

        if passed_checks:
            download_successful = True

            if self.delete_old_data:
                try:
                    shutil.rmtree(self.data_target)
                except:
                    pass
                os.makedirs(self.data_target)

            step = 0
            writer = SummaryWriter(log_dir="logs/download/s3")
            download_started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log["download_started"] = download_started
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

                                # Calculate duration and GB/s
                                file_size_mb = cameras_dict[camera]["aws-sources"][
                                    aws_source
                                ][date][file_name]["size"] / (1024**2)
                                time_end = time.time()
                                duration = time_end - time_start
                                mb_per_s = file_size_mb / duration
                                mb_per_s_list.append(mb_per_s)

                                # Update stats
                                step += 1
                                pbar.update(1)
                                writer.add_scalar(
                                    "download/mb_per_second", mb_per_s, step
                                )
            writer.close()
            download_ended = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log["download_ended"] = download_ended
            self.log["download_speed_avg_mbs"] = sum(mb_per_s_list) / len(mb_per_s_list)

        else:
            download_successful = False
            logging.error("Safety checks failed. Not downloading data")

        return download_successful

    def _mcity_check_downloaded_data(self, cameras_dict):
        downloaded_files = os.listdir(self.data_target)
        download_successful = True

        for camera in tqdm(cameras_dict, desc="Checking downloaded data"):
            for aws_source in cameras_dict[camera]["aws-sources"]:
                for date in cameras_dict[camera]["aws-sources"][aws_source]:
                    for file in cameras_dict[camera]["aws-sources"][aws_source][date]:
                        file_name = os.path.basename(file)
                        if file not in downloaded_files:
                            download_successful = False
                            logging.error(f"File {file} was not downloaded properly")
                        else:
                            cameras_dict[camera]["aws-sources"][aws_source][date][
                                file_name
                            ]["download_successful"] = True
        return download_successful

    def _mcity_set_v51_metadata(self):
        # Prepare and save metadata.json to import V51 dataset of type fo.types.FiftyoneDataset
        sample_fields = [
            {
                "name": "filepath",
                "ftype": "fiftyone.core.fields.StringField",
                "embedded_doc_type": None,
                "subfield": None,
                "fields": [],
                "db_field": "filepath",
                "description": None,
                "info": None,
                "read_only": True,
                "created_at": {"$date": "2024-11-14T15:24:21.719Z"},
            },
            {
                "name": "sensor",
                "ftype": "fiftyone.core.fields.StringField",
                "embedded_doc_type": None,
                "subfield": None,
                "fields": [],
                "db_field": "sensor",
                "description": None,
                "info": None,
                "read_only": True,
                "created_at": {"$date": "2024-11-14T15:24:21.719Z"},
            },
            {
                "name": "timestamp",
                "ftype": "fiftyone.core.fields.DateTimeField",
                "embedded_doc_type": None,
                "subfield": None,
                "fields": [],
                "db_field": "timestamp",
                "description": None,
                "info": None,
                "read_only": True,
                "created_at": {"$date": "2024-11-14T15:24:21.719Z"},
            },
        ]

        # Get version of current V51 package
        package_name = "fiftyone"
        version = pkg_resources.get_distribution(package_name).version
        version_str = str(version)

        v51_metadata = {}
        v51_metadata["version"] = version_str
        v51_metadata["sample_fields"] = sample_fields

        # Store metadata.json
        file_path = os.path.join(self.storage_target_root, "metadata.json")
        with open(file_path, "w") as json_file:
            json.dump(v51_metadata, json_file)

    def _mcity_unpack_data(self, cameras_dict):
        v51_samples = {}
        v51_samples_array = []
        for camera in tqdm(cameras_dict, desc="Unpacking data"):
            for aws_source in cameras_dict[camera]["aws-sources"]:
                for date in cameras_dict[camera]["aws-sources"][aws_source]:
                    for file in cameras_dict[camera]["aws-sources"][aws_source][date]:
                        file_name = os.path.basename(file)
                        file_path = os.path.join(self.data_target, file)
                        unpacking_successful = True
                        with open(file_path, "r") as file:
                            for line in file:
                                try:
                                    data = json.loads(line)
                                    v51_sample = {}
                                    if "time" in data and "data" in data:
                                        # Get data
                                        timestamp = data.get("time")
                                        image_base64 = data.get("data")

                                        # Get timestamp
                                        time_obj = datetime.datetime.strptime(
                                            timestamp, "%Y-%m-%d %H:%M:%S.%f"
                                        )
                                        formatted_time = (
                                            time_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")[
                                                :-3
                                            ]
                                            + "Z"
                                        )

                                    elif (
                                        "image" in data
                                        and "sensor_name" in data
                                        and "event_timestamp" in data
                                    ):
                                        # Get data
                                        image_base64 = data.get("image")
                                        sensor_name = data.get("sensor_name")
                                        timestamp = data.get("event_timestamp")

                                        # Get timestamps in UTC and Michigan time
                                        utc_time = datetime.datetime.fromtimestamp(
                                            timestamp, tz=datetime.timezone.utc
                                        )
                                        michigan_tz = pytz.timezone("America/Detroit")
                                        michigan_time = utc_time.astimezone(michigan_tz)
                                        formatted_time = (
                                            michigan_time.strftime(
                                                "%Y-%m-%dT%H:%M:%S.%f"
                                            )[:-3]
                                            + "Z"
                                        )
                                        # TODO Make sure that the conversion to Michigan time is correct
                                    else:
                                        unpacking_successful = False
                                        logging.error(
                                            f"Format cannot be processed: {data}"
                                        )
                                        continue

                                    if image_base64 and formatted_time:
                                        # Decode the base64 image data
                                        image_data = base64.b64decode(image_base64)

                                        # File paths
                                        image_filename = (
                                            f"{camera}_{formatted_time}.jpg"
                                        )
                                        output_path = os.path.join(
                                            self.data_target, image_filename
                                        )

                                        # Ensure correct timestamp format
                                        milliseconds = formatted_time.split(".")[1][
                                            :3
                                        ].ljust(3, "0")
                                        formatted_time = (
                                            formatted_time.split(".")[0]
                                            + "."
                                            + milliseconds
                                            + "Z"
                                        )
                                        iso8601_regex = re.compile(
                                            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$"
                                        )
                                        iso8601_conform = bool(
                                            iso8601_regex.match(formatted_time)
                                        )
                                        if not iso8601_conform:
                                            logging.warning(
                                                f"Timestamp does not conform to ISO8601: {formatted_time}"
                                            )

                                        # Prepare import with V51
                                        v51_sample["filepath"] = output_path
                                        v51_sample["sensor"] = camera
                                        v51_sample["timestamp"] = {
                                            "$date": formatted_time
                                        }
                                        v51_samples_array.append(v51_sample)

                                        # Save the decoded image data as a JPEG
                                        with open(output_path, "wb") as image_file:
                                            image_file.write(image_data)
                                    else:
                                        unpacking_successful = False
                                        logging.error(
                                            f"There was an issue during file processing of {file}"
                                        )
                                        continue

                                except json.JSONDecodeError as e:
                                    unpacking_successful = False
                                    logging.error(f"Error decoding JSON: {e}")

                        # Delete the original file if unpacking was successful
                        if unpacking_successful:
                            cameras_dict[camera]["aws-sources"][aws_source][date][
                                file_name
                            ]["unpack_successful"] = True
                            os.remove(file_path)

        # Save samples.json to import V51 dataset of type fo.types.FiftyoneDataset
        v51_samples["samples"] = v51_samples_array
        file_path = os.path.join(self.storage_target_root, "samples.json")
        with open(file_path, "w") as json_file:
            json.dump(v51_samples, json_file)

    def _mcity_create_v51_dataset(self, delete_old_dataset=False):
        # Delete old datasets
        if delete_old_dataset:
            for dataset_name in fo.list_datasets():
                if "data_engine_rolling" in dataset_name:
                    fo.delete_dataset(dataset_name)

        # Create the dataset
        dataset = fo.Dataset(name=self.name, overwrite=delete_old_dataset)

        dataset.add_dir(
            dataset_dir=self.storage_target_root,
            dataset_type=fo.types.FiftyOneDataset,
            progress=True,
        )

        dataset.compute_metadata(num_workers=NUM_WORKERS, progress=True)
        self.log["v51_loading_successful"] = True
        return dataset

    def _process_aws_result(self, result):
        # Get list of folders from AWS response
        if "CommonPrefixes" in result:
            folders = [prefix["Prefix"] for prefix in result["CommonPrefixes"]]
            return folders
        else:
            return None
