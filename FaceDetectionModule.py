import cv2
import mediapipe as mp
import time


# def __init__

class FaceDetector():
    def __init__(self, minDetectionCon=.9):
        self.minDetectionCon = minDetectionCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        # print(results.detections)
        # bg=cv2.imread('bg2.png')
        bboxs = []

        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin *
                                               ih), int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id, bbox, detection.score])
                # print(detection)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-15),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img, bboxs


def main():
    pTime = 0;cTime = 0
    # cap = cv2.VideoCapture('video/1.mp4')
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(minDetectionCon=0.9)
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while (True):
        ret, img = cap.read()
        if not ret:
            print("Can't receive img (stream end?). Exiting ...")
            break
        img, bboxs=detector.findFaces(img)
        print(bboxs)

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