from flask import Flask,request,render_template,url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

app=Flask(__name__)

# Load the model
model=load_model("facial_emotion_detection_model.h5")
class_names=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

UPLOAD_FOLDER='static/uploads'
# Ensure the upload directory exists
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

    # Generate a unique filename for the plot to avoid caching issues and overwriting
    plot_filename = f"{uuid.uuid4().hex}.png"
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)

    # Use 'Agg' backend for Matplotlib when not displaying interactively (e.g., in a server environment)
    plt.switch_backend('Agg')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Emotion: {predicted_class} ({confidence}%)')
    plt.axis('off') # Hide axes
    plt.savefig(plot_path)
    plt.close() # Close the plot to free up memory

    return predicted_class, confidence, plot_path

@app.route("/",methods=['GET','POST'])
def index():
    if request.method=='POST':
        if 'file' not in request.files:
            return 'No file part in the request', 400 # Return with status code
        file=request.files['file']
        if file.filename=='':
            return 'No selected file', 400 # Return with status code

        if file:
            # Generate a unique filename to prevent overwriting if multiple users upload same filename
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_path=os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            emotion, confidence, plot_path = detect_emotion(file_path)
            
            # Make plot_path relative to static folder for Flask url_for
            display_plot_path = url_for('static', filename='uploads/' + os.path.basename(plot_path))
            display_image_path = url_for('static', filename='uploads/' + os.path.basename(file_path))

            return render_template('index.html', image_path=display_image_path, emotion=emotion, confidence=confidence, plot_path=display_plot_path)

    return render_template('index.html')

if __name__=="__main__":
    # Get the port from the environment variable (Render sets this)
    # Default to 5000 for local development if PORT env var is not set
    port = int(os.environ.get('PORT', 5600))
    # Bind to 0.0.0.0 to make it accessible from outside the container
    # Set debug=False for production deployments on Render
    app.run(debug=False, host='0.0.0.0', port=port)