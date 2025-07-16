import cv2
import numpy as np
from keras.models import load_model

# Load the actual trained model
model = load_model("best_emotion_model.keras")

# Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start the webcam
webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        try:
            face = cv2.resize(face, (48, 48))
            face = extract_features(face)
            pred = model.predict(face)
            label_index = pred.argmax()
            prediction_label = labels[label_index]

            cv2.putText(frame, prediction_label, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        except Exception as e:
            print("Error:", e)

    cv2.imshow("Real-time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
