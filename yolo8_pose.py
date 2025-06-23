from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")  # or yolov8s-pose.yaml for a larger model

model.train(
    data="pose_data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="pose_keypoint_train",
    verbose=True
)
