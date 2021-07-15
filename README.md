# Emotion_detection

I have built a model which can detect your facial emotion using your webcam or photo.

# Model

***##Dataset info:***
RangeIndex: 35887 entries, 0 to 35886 Data columns (total 3 columns):

|  Column               | Non-null Count               |Dtype                          |
|----------------|-------------------------------|-----------------------------|
|emotion         |`35887 non-null`              |'int64'            |
|pixels          |`35887 non-null`              |object             |
|Usage           |`35887 non-null`              |object
dtypes: int64(1), object(2) memory usage: 841.2+ KB
[

> One of the image of Dataset displayed using matplotlib.imshow(Grayscale).


![One of the dataset image displayed using matplotlib.imshow(Grayscale)](data:image/png;base64,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)
***Emotions used in the dataset:***
'Anger',  'Disgust',  'Fear',  'Happiness',  'Sadness',  'Surprise',  'Neutral'.
**Model Training and Testing**
Model have trained using CNN architecture and achieve an accuracy of 99.10 percent with a val_accuracy: 62.08 percent.

We have used keras, Tensorflow, Numpy for Model generation and training.

## Model Deployed

Model has been deployed with Streamlit.


