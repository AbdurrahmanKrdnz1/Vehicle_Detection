import cv2
from test import model_best_weights


# In here, we will determine our paths for video records
video_path = r"C:\Users\abdur\Desktop\Vehicle_Detection\test_video_record.mp4"
out_path = r"C:\Users\abdur\Desktop\Images_Dataset\output_video.mp4"

# Open video files with opencv library
video = cv2.VideoCapture(video_path)

# Take attributes of video
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Write out video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# Assign a color for each class
class_colors = {
    0 : [0, 255, 0],
    1 : [0, 0, 255]
}

# Process video frame by frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Predict on each frame
    results = model_best_weights.predict(frame, conf = 0.5)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{model_best_weights.names[int(class_id)]} {conf:.2f}"

            # Select a color according to class
            color = class_colors.get(class_id, (255, 0, 0))

            # Draw bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # Write label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write out video
    out.write(frame)

    # Release resources after video is over
video.release()
out.release()