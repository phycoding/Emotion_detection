import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import keras
from keras.preprocessing.image import img_to_array
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

# You can load your model by training or by using the file location of downloaded model

my_model = keras.models.load_model(r'C:\Users\ASUS\stapp\my_model\my_model.h5')
lookup = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')

def image_input():
    content_file=None
    if st.sidebar.checkbox('Upload'):
        content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    else:
      st.markdown('Please select **Upload button**.')
       
    if content_file is not None:
        content = detect_face(content_file)
        x = detect_emotion(content)
        #content = np.array(content) #pil to cv
        #content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()
    return st.image(content,caption = "You are looking like to have in: %s" %x)
def detect_face(images):
  try:
    images = PIL.Image.open(images)
    new_img = np.array(images.convert('RGB'))
    faces = face_cascade.detectMultiScale(new_img, 1.1, 4)
    for (x,y,w,h) in faces:
      cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),7)
    return new_img
  except:
    #new_img = np.array(images.convert('RGB'))
    images = np.array(images)
    images.astype(np.float32)
    faces = face_cascade.detectMultiScale(images, 1.1, 4)
    for (x,y,w,h) in faces:
      cv2.rectangle(images,(x,y),(x+w,y+h),(255,0,0),7)
    return images
def webcam_input():
    st.header("Webcam Live Feed")
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([], channels='BGR')
    SIDE_WINDOW = st.sidebar.image([], width=100, channels='BGR')
    status_text = st.empty()
    
    camera = cv2.VideoCapture(0)
     

    while run:
       _, frame = camera.read()
       frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       orig = frame.copy()
       orig = detect_face(orig)
       orig1 = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
       x = detect_emotion(orig)
       FRAME_WINDOW.image(orig)
       SIDE_WINDOW.image(frame.copy())
       status_text.text('You are in: %s'%x)
       print(x)
       
    else:
        st.warning("NOTE: Streamlit currently doesn't support webcam. So to use this, clone this repo and run it on local server.")
        st.warning('Stopped')
def detect_emotion(images):
  try:
    images = PIL.Image.open(images)
    new_img = np.array(images.convert('RGB'))
    faces = face_cascade.detectMultiScale(new_img, 1.1, 4)
    for (x,y,w,h) in faces:
      gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
      roi_gray=gray[y:y+h,x:x+w]
      roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
      if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        preds=my_model.predict(roi)[0]
        
  except:
    #new_img = np.array(images.convert('RGB'))
    images = np.array(images)
    images.astype(np.float32)
    faces = face_cascade.detectMultiScale(images, 1.1, 4)
    status_text = st.empty()
    for (x,y,w,h) in faces:
      gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
      roi_gray=gray[y:y+h,x:x+w]
      status_text.empty()
      roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
      if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        preds=my_model.predict(roi)[0]
        label = lookup[np.argmax(preds)]
        return label
def main():
  st.title("Emotion Detector")
  st.sidebar.title('Navigation')

  method = st.sidebar.radio('Go To ->', options=['Image','Webcam'],key = 'amaifkajlfj')
  st.sidebar.header('Options')


  if method == 'Image':
    image_input()
  else:
    webcam_input()
if __name__ == "__main__": 
    main()
