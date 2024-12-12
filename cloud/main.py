from aws_stream_filter_framerate import SampleTimestamps
from s3_file_list import AwsDownloader
import datetime


def main():
    print('Welcome')
    start_date = datetime.datetime.strptime("2023-11-19", "%Y-%m-%d")
    end_date = datetime.datetime.strptime("2023-11-19", "%Y-%m-%d")
    sample_rate = 1
    aws_downloader = AwsDownloader(
        name='test-1',
        start_date=start_date,
        end_date=end_date,
        sample_rate_hz=sample_rate,
        test_run=True
    )
    file_names = aws_downloader.get_list()
    i = 0

    # for file_name in file_names:
    # print(file_name)
    # sampler = SampleTimestamps(file_path=file_name, target_framerate_hz=1)
    # framerate_hz, timestamps, upper_bound_threshold = sampler.get_framerate()
    # valid_target_framerate = sampler.check_target_framerate(framerate_hz)
    # if valid_target_framerate:
    #   selected_indices, selected_timestamps, target_timestamps, selected_target_timestamps = sampler.sample_timestamps(timestamps, upper_bound_threshold)
    #  sampler.update_upload_file(file_name,selected_indices)
    # break

    # sampler = SampleTimestamps(file_path="/media/dbogdoll/Datasets/aws_s3_playground/sip-data-stream2-delivery-stream-2-2024-04-12-16-25-36-1b0621a6-479c-3c30-8603-0e243780bf94", target_framerate_hz=1)
    # framerate_hz, timestamps, upper_bound_threshold = sampler.get_framerate()
    # valid_target_framerate = sampler.check_target_framerate(framerate_hz)
    # if valid_target_framerate:
    #     selected_indices, selected_timestamps, target_timestamps, selected_target_timestamps = sampler.sample_timestamps(timestamps, upper_bound_threshold)
    #     sampler.update_file(selected_indices)


if __name__ == "__main__":
    main()