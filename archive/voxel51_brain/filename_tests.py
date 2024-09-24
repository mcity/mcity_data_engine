    filenames = [
        "gridsmart_ne_stage_2_2021_07_12_labeled_20201210T112220.421.jpg",  # location: gridsmart_ne_stage_2, timestamp: 2020-12-10_11-22-20
        "beal_clip1_2023-02-20_09-35-26-816402.jpg",  # location: beal_clip1, timestamp: 2023-02-20_09-35-26
        "beal_clip2_2023-02-20_09-52-23-189766.jpg"  # location: beal_clip2, timestamp: 2023-02-20_09-52-23
        "bishop_clip20_2023-02-20_18-09-58-736420.jpg",  # location: bishop_clip20, timestamp: 2023-02-20_18-09-58
        "georgetown_clip13_2023-02-20_14-04-49-518732.jpg",  # location: georgetown_clip13, timestamp: 2023-02-20_14-04-49
        "gridsmart_ne_stage_2_2021_07_12_labeled_20201210T112220.421.jpg",  # location: gridsmart_ne_stage_2, timestamp: 2020-12-10_11-22-20,
        "gridsmart_ne_stage_1_2021_05_02_labeled_2021-05-02_09-12-49-798704.jpg",  # location: gridsmart_ne_stage_1, timestamp: 2021-05-02_09-12-49
        "Huron_Plymouth-Geddes_Huron_gs_Geddes_Huron1__2023-03-28__9__0_gs_Geddes_Huron1__2023-03-28__9__0__2023-03-28_09-07-02-239469.jpg",  # location: Huron_Plymouth-Geddes_Huron_gs_Geddes_Huron1, timestamp: 2023-03-28_09-07-02
        "Main_stadium_gs_Main_stadium1__2023-03-28__18__0_gs_Main_stadium1__2023-03-28__18__0__2023-03-28_18-53-02-023320.jpg",  # location:Main_stadium_gs_Main_stadium1, timestamp: 2023-03-28_18-53-02
    ]

    expected_results = [
        {
            "location": "gridsmart_ne_stage_2",
            "timestamp": "2020-12-10_11-22-20",
        },
        {
            "location": "beal_clip1",
            "timestamp": "2023-02-20_09-35-26",
        },
        {
            "location": "beal_clip2",
            "timestamp": "2023-02-20_09-52-23",
        },
        {
            "location": "bishop_clip20",
            "timestamp": "2023-02-20_18-09-58",
        },
        {
            "location": "georgetown_clip13",
            "timestamp": "2023-02-20_14-04-49",
        },
        {
            "location": "gridsmart_ne_stage_2",
            "timestamp": "2020-12-10_11-22-20",
        },
        {
            "location": "gridsmart_ne_stage_1",
            "timestamp": "2021-05-02_09-12-49",
        },
        {
            "location": "Huron_Plymouth-Geddes_Huron_gs_Geddes_Huron1",
            "timestamp": "2023-03-28_09-07-02",
        },
        {
            "location": "Main_stadium_gs_Main_stadium1",
            "timestamp": "2023-03-28_18-53-02",
        },
    ]

    for filename, expected in zip(filenames, expected_results):
        result = process_midadvrb_metadata(filename)
        assert result == expected, f"Failed for {filename}: {result} != {expected}"
        print(f"Passed for {filename}")

    for name in filenames:
        print(process_midadvrb_metadata(name))