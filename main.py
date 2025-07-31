from flask import Flask,request,render_template,url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np 
import os
import uuid
app=Flask(__name__)
model=load_model("facial_emotion_detection_model.h5")
class_names=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
UPLOAD_FOLDER='static/uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
def detect_emotion(image_path):
    img=image.load_img(image_path,target_size=(48,48),color_mode='grayscale')
    img_array=image.img_to_array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)
    prediction=model.predict(img_array)
    predicted_index=np.argmax(prediction)
    predicted_class=class_names[predicted_index]
    confidence=round(prediction[0][predicted_index]*100,2)
    plot_filename = f"{uuid.uuid4().hex}.png"
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    plt.switch_backend('Agg')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Emotion: {predicted_class} ({confidence}%)')
    plt.axis('off')
    plt.savefig(plot_path)
    plt.close()
    return predicted_class, confidence, plot_path
@app.route("/",methods=['GET','POST'])
def index():
    if request.method=='POST':
        if 'file' not in request.files:
            return 'NO file Uploaded'
        file=request.files['file']
        if file.filename=='':
            return 'NO file Uploaded'
        if file:
            file_path=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
            file.save(file_path)
            emotion, confidence, plot_path = detect_emotion(file_path)
            return render_template('index.html', image_path=file_path, emotion=emotion, confidence=confidence, plot_path=plot_path)


    return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)