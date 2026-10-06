import cv2
from ultralytics import YOLO
from datetime import datetime

# ==========================================
# SETTINGS
# ==========================================

CAMERA_URL = "http://192.0.0.4:8080/video"

MODEL_PATH = "yolov8n.pt"

# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("ERROR: Could not connect to camera")
    exit()

print("Camera connected!")
print("Vehicle tracking and counting started.")
print("Press Q to stop.")

# ==========================================
# VEHICLE CLASSES
# ==========================================

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not receive frame")
        continue

    height, width = frame.shape[:2]

    # ======================================
    # DATE AND TIME
    # ======================================

    now = datetime.now()

    date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")

    # ======================================
    # YOLO TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        conf=0.45,
        imgsz=640,
        verbose=False
    )

    # ======================================
    # COUNTERS
    # ======================================

    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    # ======================================
    # VEHICLES
    # ======================================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        track_ids = boxes.id.int().cpu().tolist()

        class_ids = boxes.cls.int().cpu().tolist()

        coordinates = boxes.xyxy.cpu().tolist()

        for track_id, class_id, box in zip(
            track_ids,
            class_ids,
            coordinates
        ):

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)

            vehicle_type = vehicle_classes[class_id]

            # ==================================
            # COUNT
            # ==================================

            if class_id == 2:
                car_count += 1

            elif class_id == 3:
                motorcycle_count += 1

            elif class_id == 5:
                bus_count += 1

            elif class_id == 7:
                truck_count += 1

            # ==================================
            # CENTER POINT
            # ==================================

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ==================================
            # DRAW VEHICLE BOX
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # ==================================
            # TRACKING ID
            # ==================================

            cv2.putText(
                frame,
                f"{vehicle_type} | ID {track_id}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

            # ==================================
            # CENTER POINT
            # ==================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

    # ==========================================
    # TOTAL
    # ==========================================

    total = (
        car_count
        + motorcycle_count
        + bus_count
        + truck_count
    )

    # ==========================================
    # DASHBOARD
    # ==========================================

    cv2.rectangle(
        frame,
        (0, 0),
        (350, 170),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "LIVE VEHICLE MONITOR",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycle_count}",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {bus_count}",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (15, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL: {total}",
        (15, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # ==========================================
    # DATE / TIME
    # ==========================================

    cv2.putText(
        frame,
        f"{date}  {current_time}",
        (width - 300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "Vehicle Tracking and Counting",
        frame
    )

    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Vehicle tracking stopped.")