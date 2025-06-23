import cv2
from ultralytics import YOLO

# Load model
model = YOLO("/home/dataengine/Downloads/mcity_data_engine-1/output/models/ultralytics/mcity_2844_clean_crowd_updated/yolo12x/weights/best.pt")

# Run tracking in stream mode
results = model.track(source="wheeler1_gs_Geddes_Huron_2_short (1).mp4", stream=True, tracker="bytetrack.yaml")

# Display results frame-by-frame with custom delay
for r in results:
    frame = r.plot()  # Annotated frame (NumPy array)
    cv2.imshow("Slow Playback Tracking", frame)

    # Add delay (e.g., 60ms per frame = ~16 FPS playback)
    if cv2.waitKey(250) & 0xFF == ord("q"):  # Press 'q' to quit early
        break

cv2.destroyAllWindows()
