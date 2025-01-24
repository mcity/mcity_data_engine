import base64
import datetime
import json
import logging
import os
import re
import shutil
import time

from queue import Empty
from tqdm import tqdm

import torch.multiprocessing as mp
from importlib.metadata import version as get_version

import boto3
import fiftyone as fo
import pkg_resources
import pytz
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import NUM_WORKERS

import logging

class AwsDownloader:

    def __init__(self, bucket, prefix, download_path, test_run):
        with open(".secret", "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                os.environ[key] = value

        # S3 Client
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", None),
        )

        self.bucket = bucket
        self.prefix = prefix
        self.test_run = test_run

        self.download_path = download_path
        os.makedirs(download_path, exist_ok=True)

    # Internal functions

    def _list_files(self, bucket, prefix):
        files = []
        download_size_bytes = 0
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
                    download_size_bytes += obj['Size']
        
        total_size_tb = download_size_bytes / (1024**4)

        return files, total_size_tb

    def _set_v51_metadata(self, output_folder_root):
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
                "created_at": {"$date": "2024-11-14T15:24:21.719Z"},    # FIXME Replace with actual date of function call
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
        version = get_version(package_name)
        version_str = str(version)

        v51_metadata = {}
        v51_metadata["version"] = version_str
        v51_metadata["sample_fields"] = sample_fields

        # Store metadata.json
        file_path = os.path.join(output_folder_root, "metadata.json")
        with open(file_path, "w") as json_file:
            json.dump(v51_metadata, json_file)

    def _process_file(self, file_path, output_folder_data):
        v51_samples = []

        # Prepare ISO check for time format
        iso8601_regex = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$")

        with open(file_path, "r") as file:
            for line in file:                    
                try:
                    data = json.loads(line)
                    if "time" in data and "data" in data:
                        # Get data
                        timestamp = data.get("time")
                        image_base64 = data.get("data")

                        # Get timestamp
                        try:
                            # Time with ms (default)
                            time_obj = datetime.datetime.strptime(
                                timestamp, "%Y-%m-%d %H:%M:%S.%f"
                            )
                        except ValueError:
                            time_obj = datetime.datetime.strptime(
                                timestamp, "%Y-%m-%d %H:%M:%S"
                            )
                        formatted_time = (
                            time_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")[
                                :-3
                            ]
                            + "Z"
                        )
                        sensor_name = data.get("sensor_name")

                    elif (
                        "image" in data
                        and "sensor_name" in data
                        and "event_timestamp" in data
                    ):
                        # Get data
                        sensor_name = data.get("sensor_name")
                        timestamp = data.get("event_timestamp")
                        image_base64 = data.get("image")

                        # Get timestamps in UTC and Michigan time
                        # TODO Ensure that this conversion is correct, not clear in which time timestamps were written in
                        utc_time = datetime.datetime.utcfromtimestamp(timestamp)
                        michigan_tz = pytz.timezone("America/Detroit")
                        michigan_time = utc_time.astimezone(michigan_tz)
                        formatted_time = (
                            michigan_time.strftime(
                                "%Y-%m-%dT%H:%M:%S.%f"
                            )[:-3]
                            + "Z"
                        )
                    else:
                        logging.error(f"Format cannot be processed: {data}")
                        continue

                    if image_base64 and formatted_time:

                        if sensor_name is None:         # FIXME Can be derived from S3 source if not stored in data itself
                            sensor_name = "Unknown"

                        # File paths
                        image_filename = (
                            f"{sensor_name}_{formatted_time}.jpg"
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
                        
                        iso8601_conform = bool(
                            iso8601_regex.match(formatted_time)
                        )
                        if not iso8601_conform:
                            logging.error(f"Timestamp does not conform to ISO8601: {formatted_time}")

                        # Store image to disk
                        output_path = os.path.join(
                            output_folder_data, image_filename
                        )

                        if os.path.exists(output_path):                    
                            logging.debug(f"File already exists: {output_path}")
                        else:
                            image_data = base64.b64decode(image_base64)
                            with open(output_path, "wb") as image_file:
                                image_file.write(image_data)

                        # Prepare import with V51
                        v51_sample = {
                                "filepath": output_path,
                                "sensor": sensor_name,
                                "timestamp": {"$date": formatted_time}
                            }
                        v51_samples.append(v51_sample)

                    else:
                        logging.error(f"There was an issue during file processing of {file_path}. Issues with image_base64: {image_base64 is None}, formatted_time: {formatted_time is None}, sensor_name: {sensor_name is None}")
                        continue

                except json.JSONDecodeError as e:
                    logging.error(f"File {os.path.basename(file_path)} - Error decoding JSON: {e}")       

        return v51_samples

    # External functions

    def download_files(self, log_dir, MAX_SIZE_TB=1.5):
        files_to_be_downloaded, total_size_tb = self._list_files(self.bucket, self.prefix)

        writer = SummaryWriter(log_dir=log_dir)

        n_downloaded_files, n_skipped_files = 0,0
        if total_size_tb <= MAX_SIZE_TB:
            for file in tqdm(files_to_be_downloaded, desc="Downloading files from AWS."):
                time_start = time.time()
                file_path = os.path.join(self.download_path, file)
                if not os.path.exists(file_path):
                    if self.test_run == False:
                        self.s3.download_file(self.bucket, file, file_path)  # FIXME Activate download
                    n_downloaded_files += 1

                    time_end = time.time()
                    duration = time_end - time_start
                    files_per_second = 1/duration

                    writer.add_scalar("download/files_per_second", files_per_second, n_downloaded_files)
                else:
                    logging.warning(f"Skipping {file}, already exists.")
                    n_skipped_files += 1
        else:
            logging.error(f"Total size of {total_size_tb:.2f} TB exceeds limit of {MAX_SIZE_TB} TB. Skipping download.")

        logging.info(f"Downloaded {n_downloaded_files} files, skipped {n_skipped_files} files.")

        # Check if all files were downloaded properly
        DOWNLOAD_NUMBER_SUCCESS, DOWNLOAD_SIZE_SUCCESS = False, False

        subfolders = [f.path for f in os.scandir(self.download_path) if f.is_dir()] # Each subfolder stands for a requested sample rate. Defaults to "1"
        if len(subfolders) > 0:
            sub_folder = os.path.basename(subfolders[0])
            if len(subfolders) > 1:
                logging.warning(f"More than one subfolder in download directory {self.download_path} found, selected {sub_folder}.")

            downloaded_files = []
            downloaded_size_bytes = 0

            folder_path = os.path.join(self.download_path, sub_folder)  
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            with tqdm(total=len(files), desc="Checking downloaded files", unit="file") as pbar:
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    downloaded_files.append(file_path)
                    downloaded_size_bytes += os.path.getsize(file_path)
                    pbar.update(1)

            downloaded_size_tb = downloaded_size_bytes / (1024**4)
            logging.info(f"Downloaded {len(downloaded_files)} files. Total size: {downloaded_size_tb:.2f} TB")

            if len(files_to_be_downloaded) == len(downloaded_files):
                logging.info("All files downloaded successfully.")
                DOWNLOAD_NUMBER_SUCCESS = True
            else:
                logging.error(f"Only {len(downloaded_files)} of {len(files_to_be_downloaded)} planned files downloaded.")
                DOWNLOAD_NUMBER_SUCCESS = False

            if total_size_tb == downloaded_size_tb:
                logging.info("The downloaded size equals the planned download size.")
                DOWNLOAD_SIZE_SUCCESS = True
            else:
                logging.error(f"The downloaded size of {downloaded_size_tb} TB varies from the planned download size of {total_size_tb} TB.")
                DOWNLOAD_SIZE_SUCCESS = False
        else:
            sub_folder = None
            logging.error(f"No subfolder found in download directory {self.download_path}.")

        writer.close()

        return sub_folder, files, DOWNLOAD_NUMBER_SUCCESS, DOWNLOAD_SIZE_SUCCESS

    # Decode data with multiple workers
    def decode_data(self, sub_folder, files, log_dir, dataset_name = "annarbor_rolling", output_folder = "decoded", dataset_persistance=True):
        
        output_folder_root = os.path.join(self.download_path, sub_folder, output_folder)
        output_folder_data = os.path.join(output_folder_root, "data")
        os.makedirs(output_folder_data, exist_ok=True)

        # Save metadata to import dataset as Voxel51 dataset
        self._set_v51_metadata(output_folder_root)

        # Extract file content
        json_file_path = os.path.join(output_folder_root, "samples.json")
        
        # Queue for multiprocessing results
        result_queue = mp.Queue()
        task_queue = mp.Queue()

        # Add files to the task queue
        for file_path in files:
            absolute_file_path = os.path.join(self.download_path, sub_folder, file_path)
            task_queue.put(absolute_file_path)

        logging.info(f"Added {len(files)} files to task queue. Ready for multiprocessing.")

        # Gather events for extraction workers 
        worker_done_events = []
        n_extraction_workers = NUM_WORKERS
        for _ in range(n_extraction_workers):
            done_event = mp.Event()
            worker_done_events.append(done_event)

        # Start the result worker
        result_worker_process = mp.Process(target=self.result_json_worker, args=(result_queue, json_file_path, worker_done_events, log_dir))
        result_worker_process.start()

        # Start the data extraction workers
        n_files_per_worker = int(len(files) / n_extraction_workers)
        workers = []
        for done_event in worker_done_events:
            p = mp.Process(target=self.data_extraction_worker, args=(task_queue, result_queue, done_event, output_folder_data, n_files_per_worker))
            p.start()
            workers.append(p)

        # Waiting for data extraction workers
        for p in workers:
            p.join()
        logging.info("All data processing workers finished processing.")

        # Waiting for data processing worker
        result_worker_process.join()
        logging.info("JSON worker finished processing.")

        task_queue.close()
        result_queue.close()

        # Load V51 dataset
        dataset = fo.Dataset(name=dataset_name)
        dataset.add_dir(
            dataset_dir=output_folder_root,
            dataset_type=fo.types.FiftyOneDataset,
            progress=True,
        )

        dataset.compute_metadata(num_workers=NUM_WORKERS, progress=True)
        dataset.persistent = dataset_persistance

        return dataset


    # Worker functions
    def data_extraction_worker(self, task_queue, result_queue, done_event, output_folder_data, n_files_per_worker):
        
        logging.info(f"Process ID: {os.getpid()}. Data extraction process started.")    

        n_files_processed = 0
        n_samples_processed = 0
        while True:
            if task_queue.empty() == True:  # If queue is empty, break out
                break
            else:
                try:
                    file_path = task_queue.get(timeout = 0.1)  # Get a file path from the task queue
                    v51_samples = self._process_file(file_path, output_folder_data)
                    if len(v51_samples) > 0:
                        result_queue.put(v51_samples)
                        if n_files_processed % 100 == 0:
                            logging.info(f"Worker {os.getpid()} finished {n_samples_processed} samples. {n_files_processed} / ~{n_files_per_worker} files done.")
                        n_files_processed += 1
                        n_samples_processed += len(v51_samples)

                except Exception as e:
                    logging.error(f"Error occured during processing of file {os.path.basename(file_path)}: {e}")
                    continue

        # Once all tasks are processed, set the done_event
        done_event.set()
        logging.info(f"Data Extraction worker {os.getpid()} shutting down.")
        return True

    def result_json_worker(self,result_queue, json_file_path, worker_done_events, log_dir):
        # Check if the samples.json file exists and load existing data if it does
        logging.info(f"Process ID: {os.getpid()}. Results processing process started.")    

        writer = SummaryWriter(log_dir=log_dir)
        n_files_processed = 0
        n_images_extracted = 0

        v51_samples_dict = {"samples": []}

        # Update the file with incoming results as long as workers are running
        while len(worker_done_events) > 0 or not result_queue.empty():
            try:
                v51_samples = result_queue.get(timeout=1)
                v51_samples_dict["samples"].extend(v51_samples)

                n_images_extracted += len(v51_samples)
                writer.add_scalar("decode/samples", n_images_extracted, n_files_processed)
                n_files_processed += 1

                # Log every 100,000 processed samples
                if n_files_processed % 1_000 == 0:
                    logging.info(f"{n_images_extracted} samples added to dict. Items in queue: {result_queue.qsize()}. Active workers: {len(worker_done_events)}")

            except Empty:
                # Empty queue is expected sometimes
                pass

            except Exception as e:
                logging.error(f"JSON worker error: {e}")
                continue

            # Check if any worker is done
            for event in list(worker_done_events):
                if event.is_set():
                    worker_done_events.remove(event)

        # Store data to JSON
        logging.info(f"Storing data to samples.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(v51_samples_dict, json_file)

        writer.close()
        logging.info(f"JSON worker {os.getpid()} shutting down.")
        return True