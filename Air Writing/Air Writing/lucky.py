import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    ret, frame = cap.read()
    H, W, _ = frame.shape

    # Canvases
    canvas_black = np.zeros((H, W, 3), dtype=np.uint8)   # for overlay with camera
    canvas_white = np.ones((H, W, 3), dtype=np.uint8) * 255  # clean canvas (only shows after 'L')

    thickness = 8
    last_point = None

    # HSV color range for blue pen
    lower_color = np.array([100, 150, 50])
    upper_color = np.array([140, 255, 255])

    # Fullscreen camera window
    cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Air Writing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("[INFO] Controls: L = show canvas, C = clear, Q = quit")

    canvas_visible = False  # only show canvas after 'L'

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
            cv2.line(canvas_black, last_point, center, (255,255,255), thickness, cv2.LINE_AA)
            last_point = center
        else:
            last_point = None

        # Overlay drawing on live feed
        combined = cv2.addWeighted(frame, 0.7, canvas_black, 0.3, 0)
        cv2.imshow("Air Writing", combined)

        if canvas_visible:
            cv2.imshow("Canvas", canvas_white)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # quit
            break
        elif key == ord('c'):  # clear everything
            canvas_black[:] = 0
            canvas_white[:] = 255
            canvas_visible = False
            cv2.destroyWindow("Canvas")
        elif key == ord('l'):  # show strokes on clean white canvas
            gray = cv2.cvtColor(canvas_black, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            canvas_white = 255 - th_bgr
            if not canvas_visible:
                cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
                canvas_visible = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
