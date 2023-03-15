import cv2
import mediapipe as mp
import time

pTime = 0
cap = cv2.VideoCapture('video/3.mp4')
# cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec((0,255,0),thickness=1,circle_radius=1)
drawSpecLine = mpDraw.DrawingSpec((255,255,255),thickness=2,circle_radius=1)

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
    img = cv2.flip(img,1)
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec,drawSpecLine)

    k = cv2.waitKey(1)
    cv2.imshow('face Mesh', img)

    # press q key to close the program
    if k & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()