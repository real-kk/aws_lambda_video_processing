import json
import boto3
import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import uuid
from urllib.parse import unquote_plus
import os
import requests
import key

s3 = boto3.client('s3', region_name="ap-northeast-2")

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

def cv2_processing(key, download_path):

    KEY = key.KEY
    ENDPOINT = key.ENDPOINT
    
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    VIDEO_FILE_PATH = download_path
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    
    if cap.isOpened() == False:
        print ('Can\'t open the video (%d)' % (VIDEO_FILE_PATH))
        exit()
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    filename = '/tmp/output_video.avi'
    out = cv2.VideoWriter(filename, fourcc, fps, (int(height), int(width)))
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load('haarcascade_frontalface_alt.xml')
    
    cnt = 0
    tmp_rectangle = None
    tmp_emotion_str = None
    while True:
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
        cv2.imwrite('/tmp/frame.jpg',frame)
        with open('/tmp/frame.jpg', "rb") as face_fd:
            detected_faces = face_client.face.detect_with_stream(face_fd, return_face_attributes=["emotion"])
            if not detected_faces and tmp_emotion_str and tmp_rectangle:
                drawRectangleAndText(frame, tmp_emotion_str, tmp_rectangle)
                out.write(frame)
                continue
            rectangle = getRectangle(detected_faces[0])
            emotion_str = ''
            emotions = list(map(str, str(detected_faces[0].face_attributes.emotion).replace(' ', '').replace('\'', '').rstrip('}').lstrip('{').split(',')[1:]))
            for emotion in emotions:
                emotion_str += emotion + '\n'
                emotion_name, emotion_score = emotion.split(":")
                emotion_name = emotion_name.strip('\'')
                emotion_score = float(emotion_score)
                emotion_dict[emotion_name] = emotion_dict.get(emotion_name, 0) + emotion_score
            emotion_count += 1
            drawRectangleAndText(frame, emotion_str, rectangle)
            tmp_rectangle = rectangle
            tmp_emotion_str = emotion_str
        if cv2.waitKey(25) == 13:
            break
        
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

emotion_dict = dict()
emotion_count = 0

def lambda_handler(event, context):
    
    bucket = event["Records"][0]["s3"]["bucket"]["name"] 
    key = unquote_plus(event["Records"][0]["s3"]["object"]["key"]) 
    download_path="/tmp/"+key

    s3.download_file(bucket, key, download_path)   

    result_file_path = cv2_processing(key, download_path)
    video_file_path = '/tmp/output_video.avi'
    s3.upload_file(video_file_path, 'processed-video-lambda', key, ExtraArgs={'ACL': 'public-read'})
    
    for emotion in emotion_dict:
            emotion_dict[emotion] /= emotion_count
        url = 'http://ec2-13-209-32-113.ap-northeast-2.compute.amazonaws.com/tasks/sentiment/'
        data = {'sentiments' : str(emotion_dict), 'key' : key}
        response = requests.post(url=url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        print(response.status_code)

    return {}


