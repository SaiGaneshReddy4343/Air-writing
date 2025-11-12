import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

MODEL_PATH = "emnist_letters_fast.keras"

# ---------------------------------------------------
# Pre-trained lightweight model (download once)
# ---------------------------------------------------
def load_model_fast():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)

    # Small CNN, trained quickly on EMNIST Letters (weights assumed pre-downloaded)
    # For first run, we build + random weights (won't predict well).
    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(26, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("[INFO] Model created but no pre-trained weights found.")
    return model

# ---------------------------------------------------
# Preprocess drawing for prediction
# ---------------------------------------------------
def preprocess_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(th > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    roi = th[y_min:y_max+1, x_min:x_max+1]

    h, w = roi.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = roi

    small = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    inv = 255 - small

    arr = inv.astype("float32") / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]
    return arr

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
def main():
    model = load_model_fast()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    ret, frame = cap.read()
    H, W, _ = frame.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    thickness = 8
    last_point = None
    prediction = None
    show_canvas = False

    # HSV color range for pen tip (blue pen example)
    lower_color = np.array([100, 150, 50])
    upper_color = np.array([140, 255, 255])

    cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Air Writing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("[INFO] Controls: P = predict, C = clear, Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                cv2.circle(frame, center, 5, (0,255,0), -1)

        if center is not None:
            if last_point is None:
                last_point = center
            cv2.line(canvas, last_point, center, (255,255,255), thickness, cv2.LINE_AA)
            last_point = center
        else:
            last_point = None

        # Show full camera + overlay
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

        if prediction is not None:
            cv2.putText(combined, f"Prediction: {prediction[0]} ({prediction[1]*100:.1f}%)",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

        cv2.imshow("Air Writing", combined)

        if show_canvas:
            cv2.imshow("Canvas", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
            prediction = None
        elif key == ord('p'):
            arr = preprocess_canvas(canvas)
            if arr is not None:
                probs = model.predict(arr, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                conf = float(np.max(probs))
                pred_letter = chr(pred_idx + ord('A'))  # A–Z
                prediction = (pred_letter, conf)
            show_canvas = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
