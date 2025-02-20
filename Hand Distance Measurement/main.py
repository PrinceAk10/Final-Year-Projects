import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Hand Detector setup
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Polynomial coefficients for the distance-to-cm mapping
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

# Main loop
while True:
    success, img = cap.read()  # Capture frame from webcam
    if not success:
        print("Failed to capture image from webcam")
        break

    # Detect hands
    hands = detector.findHands(img, draw=False)

    if hands:  # If at least one hand is detected
        hand = hands[0]  # Access the first hand (list of landmarks)
        print(hand)       # Debugging: Inspect the hand structure

        if len(hand) > 17:  # Ensure there are enough landmarks
            # Coordinates of landmarks 5 and 17
            x1, y1 = hand[5][:2]  # Index finger base
            x2, y2 = hand[17][:2]  # Pinky finger base

            # Calculate Euclidean distance between points
            distance = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C

            # Draw bounding box and distance on the image
            bbox = detector.findHands(img, draw=False)[0]['bbox']  # Bounding box info
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10), scale=2, thickness=2)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
