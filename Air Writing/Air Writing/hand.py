import cv2
import numpy as np

def get_fingertip(contour):
    # Get the highest point of the contour (smallest y-value)
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    return topmost

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    canvas = None
    last_point = None
    thickness = 6

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert to HSV for color filtering (fingers usually skin-colored)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range (adjust if needed)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        if contours:
            # Pick contour that looks like a finger (small, thin, not face)
            finger_candidates = [c for c in contours if 2000 < cv2.contourArea(c) < 8000]
            if finger_candidates:
                c = max(finger_candidates, key=cv2.contourArea)
                fingertip = get_fingertip(c)
                if fingertip:
                    center = fingertip
                    cv2.circle(frame, center, 8, (0, 255, 0), -1)
                    cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)

        if center is not None:
            if last_point is None:
                last_point = center
            cv2.line(canvas, last_point, center, (255, 255, 255), thickness, cv2.LINE_AA)
            last_point = center
        else:
            last_point = None

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Air Writing (Finger Tracking)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
