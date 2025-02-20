import cv2
import time
import os
import HandTrackingModule as htm  # Ensure HandTrackingModule.py is in the correct directory

# Webcam resolution
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)  # Use 0 if 1 doesn't work
cap.set(3, wCam)
cap.set(4, hCam)

# Folder for finger images
folderPath = "FingerImages"
if not os.path.exists(folderPath):
    print(f"Error: Folder '{folderPath}' not found.")
    exit()

myList = os.listdir(folderPath)
print("Loaded images:", myList)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
print("Number of overlays:", len(overlayList))

# Hand detector
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(f"Total fingers: {totalFingers}")

        if totalFingers > 0 and totalFingers <= len(overlayList):
            h, w, c = overlayList[totalFingers - 1].shape
            img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
