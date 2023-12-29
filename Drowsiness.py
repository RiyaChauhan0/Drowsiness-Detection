from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
frequency = 2500
duration = 1000

def eyeAspectRatio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a+b)/(2.0*c)
    return ear

count = 0
earThresh = 0.3     #ear - Eye Aspect Ratio (Distance between both eyelids i.e, vertical eye coordinates)
earFrames = 48  #consecutive frames for eye closure (for how many frames the eyes are closed)
shapePredictor = "shape_predictor_68_face_landmarks.dat"
#shapePredictor = open(file_path, 'r')

cam = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

#to get the coordinates of left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)   #detecting the face first

    for rect in rects:                  #if face is detected then detec the eyes in it 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR)/2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,0,255),1)     #draw the contours to make outlines of the eyes
        cv2.drawContours(frame, [rightEyeHull], -1, (0,0,255),1)

        if ear < earThresh:
            count += 1

            if count >= earFrames:
                cv2.putTest(frame, "DROWSINESS DETECTED", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                windsound.Beep(frequency, duration)

            else:
                count=0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

    

        

        
