import cv2
import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import key # for secret key

def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    return ((left, top), (right, bottom))

def drawRectangleAndText(frame, emotion_str, tmp_rectangle):
    y0, dy = 70, 45
    for i, emotion in enumerate(emotion_str.split('\n')):
        y = y0 + i*dy
        cv2.putText(frame, emotion, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1, 2)
    cv2.rectangle(frame, tmp_rectangle[0], tmp_rectangle[1], (0, 0, 255), 2)

KEY = key.KEY
ENDPOINT = key.ENDPOINT

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

VIDEO_FILE_PATH =  os.path.abspath('.') + '\\test.mp4'
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

if cap.isOpened() == False:
    print ('Can\'t open the video (%d)' % (VIDEO_FILE_PATH))
    exit()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

filename = 'test.avi'
out = cv2.VideoWriter(filename, fourcc, fps, (int(height), int(width)))

face_cascade = cv2.CascadeClassifier()
face_cascade.load('haarcascade_frontalface_alt.xml')

cnt = 0
tmp_rectangle = None
while (True):
    ret, frame = cap.read()
    if width > height:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        break
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur =  cv2.GaussianBlur(grayframe,(5,5), 0)
    faces = face_cascade.detectMultiScale(blur, 1.8, 2, 0, (50, 50))

    if not list(faces) and tmp_rectangle and tmp_emotion_str:
        drawRectangleAndText(frame, tmp_emotion_str, tmp_rectangle)
        out.write(frame)
        continue

    cnt += 1
    if cnt % 5 != 0 and tmp_rectangle and tmp_emotion_str:
        out.write(frame)
        continue
    cv2.imwrite('frame.jpg',frame)
    with open(os.path.abspath('.') + '\\frame.jpg', "rb") as face_fd:
        detected_faces = face_client.face.detect_with_stream(face_fd, return_face_attributes=["emotion"])
        rectangle = getRectangle(detected_faces[0])
        if not detected_faces:
            drawRectangleAndText(frame, tmp_emotion_str, tmp_rectangle)
            out.write(frame)
            continue
        emotion_str = ''
        emotions = list(map(str, str(detected_faces[0].face_attributes.emotion).replace(' ', '').replace('\'', '').rstrip('}').lstrip('{').split(',')[1:]))
        for emotion in emotions:
            emotion_str += emotion + '\n'
        drawRectangleAndText(frame, emotion_str, rectangle)
        tmp_rectangle = rectangle
        tmp_emotion_str = emotion_str

    # cv2.imshow('Video', frame)

    if cv2.waitKey(25) == 13:
        break

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()