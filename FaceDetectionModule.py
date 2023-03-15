import cv2
import mediapipe as mp
import time


# def __init__

class FaceDetector():
    def __init__(self, minDetectionCon=0.9):
        self.minDetectionCon = minDetectionCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        # print(results.detections)
        # bg=cv2.imread('bg2.png')
        bboxs = []

        if results.detections:
            for id, detection in enumerate(results.detections):
                if detection.score[0]<0.70:
                    continue
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin *
                                               ih), int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id, bbox, detection.score])
                # print(detection)
                # cv2.rectangle(img, bbox, (0, 255, 0), 2)
                if draw:
                    img=self.fancyDraw(img,bbox)

                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-15),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        return img, bboxs

    def fancyDraw(self,img,bbox,l=15,t=2,rt=1):
        x,y,w,h=bbox
        x1,y1=x+w, y+h
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        # topLeft
        cv2.line(img,(x,y),(x+l,y),(0, 255, 0),t)
        cv2.line(img,(x,y),(x,y+l),(0, 255, 0),t)
        # topRight
        cv2.line(img,(x1,y),(x1-l,y),(0, 255, 0),t)
        cv2.line(img,(x1,y),(x1,y+l),(0, 255, 0),t)
        # bottomLeft
        cv2.line(img,(x,y1),(x+l,y1),(0, 255, 0),t)
        cv2.line(img,(x,y1),(x,y1-l),(0, 255, 0),t)
        # bottomRight
        cv2.line(img,(x1,y1),(x1-l,y1),(0, 255, 0),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(0, 255, 0),t)
        return img

def main():
    pTime = 0;cTime = 0
    # cap = cv2.VideoCapture('video/1.mp4')
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(0.75)
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while (True):
        ret, img = cap.read()
        if not ret:
            print("Can't receive img (stream end?). Exiting ...")
            break
        img = cv2.flip(img,1)
        img, bboxs=detector.findFaces(img)
        # print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        k = cv2.waitKey(1)
        cv2.imshow('face detection', img)

        # press q key to close the program
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()