import cv2
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the Hand Detector module.
        
        :param mode: Static or dynamic mode for hand detection.
        :param maxHands: Maximum number of hands to detect.
        :param detectionCon: Minimum confidence value for hand detection.
        :param trackCon: Minimum confidence value for hand tracking.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Detects hands in the image and optionally draws landmarks.

        :param img: The input image in which hands are to be detected.
        :param draw: Whether to draw hand landmarks and connections.
        :return: List of detected hands with landmarks and bounding box information.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        handsList = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))

                # Bounding Box Calculation
                xList = [lm[0] for lm in lmList]
                yList = [lm[1] for lm in lmList]
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)
                bbox = (xMin, yMin, xMax - xMin, yMax - yMin)

                handsList.append({'lmList': lmList, 'bbox': bbox})

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return handsList

    @staticmethod
    def findDistance(p1, p2, img=None, draw=True, r=10, t=2):
        """
        Finds the Euclidean distance between two points.
        
        :param p1: First point as (x, y).
        :param p2: Second point as (x, y).
        :param img: Image to draw the points and line (if draw=True).
        :param draw: Whether to draw the points and line.
        :param r: Radius of the circles to draw.
        :param t: Thickness of the line to draw.
        :return: Distance between the points and the midpoint.
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 255, 0), cv2.FILLED)

        return distance, (cx, cy)
