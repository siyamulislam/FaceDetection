import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
pTime = 0
cTime = 0
cap = cv2.VideoCapture('video/1.mp4')
# cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

while (True):
    ret, img = cap.read()
    if not ret:
        print("Can't receive img (stream end?). Exiting ...")
        break
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 60),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results.detections)
    # bg=cv2.imread('bg2.png')

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin *
                                           ih), int(bboxC.width*iw), int(bboxC.height*ih)
            print(detection)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-15),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    k = cv2.waitKey(1)
    cv2.imshow('face detection', img)

    # press q key to close the program
    if k & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
