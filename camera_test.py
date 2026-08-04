import cv2

for i in range(5):
    print(f"Trying camera {i}...")

    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera {i}: Opened={cap.isOpened()}, Frame={ret}")

        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

    cap.release()