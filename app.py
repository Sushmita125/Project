from flask import Flask, render_template, Response,jsonify
from flask_cors import cross_origin
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

#References :https://www.youtube.com/watch?v=mzX5oqd3pKA&t=524s
#Reading all the models

gendermodel=load_model(r'gender_model.h5')
agemodel=load_model(r'age_model.h5')
emotionmodel=load_model(r'Emotion_model.hdf5')

app=Flask(__name__)

# To get a video capture object for the camera
camera = cv2.VideoCapture(0)

#Generate frames for live video feed
def generate_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            facesdetected = face_haar_cascade.detectMultiScale(frame, 1.32, 5)
            #Draw the rectangle around each face
            for (x, y, w, h) in facesdetected:#checking for multiple faces
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2) #CV2 uses BGR -Blue color rectangle

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            #Return the frames to video feed function
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Function called to return frames to getAllData method if frames read from camera
def gen_image():
    while True:
        success, frame = camera.read()  # read the camera frame
        if success:
            return frame
        else:
            return

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Function to retrieve Emotion from Frame
def getEmotionsFromRoi(roi):
    emotions = []
    if roi is None:
        return emotions
    roiemotion = cv2.resize(roi, (48, 48)) #Resizing
    imagepixels1 = image.img_to_array(roiemotion)
    imagepixels1 = np.expand_dims(imagepixels1, axis=0)
    imagepixels1 /= 255  #Scaling

    emotionpred = emotionmodel.predict(imagepixels1) #Model prediction
    max_index = np.argmax(emotionpred[0])       #Taking maximum value out of emotions

    emotion = ('Angry', 'Disgusted', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised')
    predictedemotion = emotion[max_index]
    emotions.append(predictedemotion)  #Appending to an array
    return emotions

#Function to retrieve Age from Frame
def getAgesFromRoi(roi):
    ages = []
    roi = cv2.resize(roi, (200, 200))  #Resize to 200*200 as model is trained on this size
    if roi is None:
        return ages
    imagepixels = image.img_to_array(roi)
    imagepixels = np.expand_dims(imagepixels, axis=0)
    imagepixels /= 255  #Scaling

    img = imagepixels.reshape(-1, 200, 200, 3)

    age = agemodel.predict(img)  #Predicting age
    # print(int(age))
    age = int(age)
    ages.append(age) #Appending age
    return ages

#Function to get Gender from frame
def getGendersFromRoi(roi):
    genders = []
    if roi is None:
        return genders
    roi = cv2.resize(roi, (200, 200))  #Resizing image
    imagepixels = image.img_to_array(roi)
    imagepixels = np.expand_dims(imagepixels, axis=0)
    imagepixels /= 255   #Scaling the images

    predictions = gendermodel.predict(imagepixels) #Predicting gender
    #If the predictions close to 0 then Male else Female
    if predictions < 0.5:
        genders.append('Male')
    else:
        genders.append('Female')
    return genders

#To get Emotion,Age,Gender JSON together
@app.route('/getAllData')
@cross_origin()
def getAllData():
    frame =gen_image()
    #print(frame)
    #Detecting face in frame
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    facesdetected = face_haar_cascade.detectMultiScale(frame, 1.32, 5)  # calls a function in xml to detect face
    ages = []
    emotions=[]
    genders=[]

    #For each face detected on frame
    for (x, y, w, h) in facesdetected:
        roi = frame[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        ages = getAgesFromRoi(roi)
        emotions = getEmotionsFromRoi(roi)
        genders = getGendersFromRoi(roi)

    #Return JSON
    return jsonify(Age=ages, Emotion=emotions, Gender=genders)
if __name__=='__main__':
    app.run(host='0.0.0.0', port=80)
