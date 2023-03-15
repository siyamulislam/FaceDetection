import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces, self.refine_landmarks,
                                                 self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(
            (0, 255, 0), thickness=1, circle_radius=1)
        self.drawSpecLine = self.mpDraw.DrawingSpec(
            (255, 255, 255), thickness=2, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces=[]
        
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                            self.drawSpecLine)
                face=[]
                for id,lm in enumerate(faceLms.landmark):
                    ih,iw,ic =img.shape
                    x,y =int(lm.x*iw),int(lm.y*ih)
                    cv2.putText(img, str(id), (x, y),cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    face.append([x,y])
                faces.append(face)
        return img,faces


def main():
    pTime = 0
    cap = cv2.VideoCapture('video/1.mp4')
    # cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    detector = FaceMeshDetector(static_image_mode=True, max_num_faces=4, min_detection_confidence=0.1,
                                refine_landmarks=False, min_tracking_confidence=0.1)
    while (True):
        ret, img = cap.read()
        if not ret:
            print("Can't receive img (stream end?). Exiting ...")
            break
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # img = cv2.flip(img, 1)
        img,faces = detector.findFaceMesh(img, draw=False)
        # print(faces[0])
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)

        k = cv2.waitKey(1)
        cv2.imshow('face Mesh', img)

        # press q key to close the program
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()