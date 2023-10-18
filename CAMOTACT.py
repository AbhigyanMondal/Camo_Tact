from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)

detector = FaceDetector(minDetectionCon=0.5, modelSelection=1)

detector1 = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)
detector2 = HandDetector(staticMode=False,
                        maxHands=2,
                        modelComplexity=1,
                        detectionCon=0.5,
                        minTrackCon=0.5)
detector3 = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

while True:
    success, img = cap.read()

    img, bboxs = detector.findFaces(img, draw=False)
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 15),border=2)
            cvzone.cornerRect(img, (x, y, w, h))

    img1 = detector1.findPose(img)
    lmList, bboxInfo = detector1.findPosition(img1, draw=True, bboxWithHands=False)
    if lmList:
        center = bboxInfo["center"]
        cv2.circle(img1, center, 5, (255, 0, 255), cv2.FILLED)

    hands, img = detector2.findHands(img, draw=True, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1['center']
        handType1 = hand1["type"]

        fingers1 = detector2.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")

        tipOfIndexFinger = lmList1[8][0:2]
        tipOfMiddleFinger = lmList1[12][0:2]

        length, info, img = detector2.findDistance(tipOfIndexFinger,tipOfMiddleFinger , img, color=(255, 0, 255),
                                                  scale=5)
    if len(hands) == 2:
        hand2 = hands[1]
        lmList2 = hand2["lmList"]
        bbox2 = hand2["bbox"]
        center2 = hand2['center']
        handType2 = hand2["type"]

        fingers2 = detector2.fingersUp(hand2)
        print(f'H2 = {fingers2.count(1)}', end=" ")
        tipOfIndexFinger2 = lmList2[8][0:2]
        length, info, img = detector2.findDistance(tipOfIndexFinger, tipOfIndexFinger2, img, color=(255, 0, 0),
                                                  scale=10)

    img, faces = detector3.findFaceMesh(img, draw=True)
    if faces:
        for face in faces:
            leftEyeUpPoint = face[50]
            leftEyeDownPoint = face[5]
            leftEyeVerticalDistance, info = detector3.findDistance(leftEyeUpPoint, leftEyeDownPoint)
            print(leftEyeVerticalDistance)



    cv2.imshow("Image", img)
    cv2.waitKey(1)


