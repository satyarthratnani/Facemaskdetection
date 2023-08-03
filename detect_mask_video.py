import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

prototxtPath = "C:/Users/Satyarth/Desktop/Face_mask_project/Face-Mask-Detection/face_detector/deploy.prototxt"
weightsPath = "C:/Users/Satyarth/Desktop/Face_mask_project/Face-Mask-Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model('C:/Users/Satyarth/Desktop/Face_mask_project/Face-Mask-Detection/mask_detector.model')

from ffpyplayer.player import MediaPlayer
cap="C:/Users/Satyarth/PycharmProjects/pythonProject/test.mp4"


def pre_dect(frame, faceNet, model):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence >= 0.168:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    for k in faces:
        preds.append(model.predict(k))
    return (locs, preds)

def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    data = []
    font = cv2.FONT_HERSHEY_COMPLEX
    while True:
        grabbed, frame = video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        flag, frame = video.read()
        if flag:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            _, frame = video.read()

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = pre_dect(frame, faceNet, model)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                cla = np.argmax(pred[0])
                label = "Mask" if cla == 0 else "No Mask"
                color = (0, 255, 0) if cla == 0 else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(pred[0]) * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            if cv2.waitKey(1) & 0xFF == 27:
                break
            ims = cv2.resize(frame, (960, 540))
            cv2.imshow("Frame", ims)
            if val != 'eof' and audio_frame is not None:
                # audio
                img, t = audio_frame

            # do a bit of cleanup
    video.release()
    cv2.destroyAllWindows()
PlayVideo(cap)