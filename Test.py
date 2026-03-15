import cv2

from emotion_engine import EmotionEngine


def run_webcam_detector():
    engine = EmotionEngine()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, results = engine.detect(frame)
        if not results:
            cv2.putText(frame, "No Face Found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_detector()
