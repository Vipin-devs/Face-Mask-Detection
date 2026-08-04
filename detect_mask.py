import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/mask_detector.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Open Camo/webcam camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera not detected")
    exit()

print("Face Mask Detection Started...")

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to read frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize for CNN
        face = cv2.resize(face, (128, 128))

        # Normalize
        face = face.astype("float32") / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Prediction
        prediction = model.predict(face, verbose=0)[0][0]

        # Check label
        if prediction < 0.5:
            label = "Mask"
            color = (0, 255, 0)
            confidence = (1 - prediction) * 100
        else:
            label = "No Mask"
            color = (0, 0, 255)
            confidence = prediction * 100

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # Display text
        text = f"{label}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break


camera.release()
cv2.destroyAllWindows()