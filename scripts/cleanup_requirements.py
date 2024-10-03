def remove_line_from_file(file_path, line_to_remove):
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            if line.strip() != line_to_remove:
                file.write(line)


requirements_file = "requirements.txt"
line_to_remove = "-e git+https://github.com/daniel-bogdoll/nuscenes-devkit@53d606f3b20af3ba90bc6b876ffbc567fa76893c#egg=nuscenes_devkit&subdirectory=../../../data_loader/nuscenes-devkit"

remove_line_from_file(requirements_file, line_to_remove)
