import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Load dataset and extract landmarks
def extract_hand_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

def load_dataset(dataset_path, max_images_per_class=200):
    X, y = [], []
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir): continue
        count = 0
        for file in os.listdir(class_dir):
            if count >= max_images_per_class: break
            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path)
            if img is None: continue
            lm = extract_hand_landmarks(img)
            if lm is not None:
                X.append(lm)
                y.append(label)
                count += 1
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_dataset("asl_alphabet_train")
print("Training model...")

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Real-time webcam recognition
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    lm = extract_hand_landmarks(frame)
    if lm is not None:
        prediction = knn.predict([lm])[0]
        cv2.putText(frame, f'Prediction: {prediction}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()