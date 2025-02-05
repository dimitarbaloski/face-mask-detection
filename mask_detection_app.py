from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train6/weights/best.pt")


def test_images():
    image_paths = [
        "test_images/face-mask-study.jpg",
        "test_images/slika1.jpg",
        "test_images/test_image.jpg"
    ]

    for image_path in image_paths:
        image = cv2.imread(image_path)

        image = cv2.resize(image, (860, 640))

        results = model.predict(image, iou=0.4)

        result_image = results[0].plot()
        cv2.imshow(f"Mask Detection - {image_path}", result_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def test_video():
    video_path = "testVideo3.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 640))

        results = model.predict(frame, iou=0.2)

        result_frame = results[0].plot()
        cv2.imshow("Mask Detection", result_frame)

        if cv2.waitKey(9) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (800, 600))
        frame = cv2.flip(frame, 1)

        results = model.predict(frame, iou=0.4)

        result_frame = results[0].plot()

        cv2.imshow("Mask Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("1: Test images")
    print("2: Test video")
    print("3: Test camera")

    choice = input("Enter 1, 2, or 3: ")

    if choice == '1':
        test_images()
    elif choice == '2':
        test_video()
    elif choice == '3':
        test_camera()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
