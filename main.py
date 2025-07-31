from flask import Flask,request,render_template,url_for,Response,jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
import cv2
import base64
from PIL import Image
import io

app=Flask(__name__)

# Load the model
model=load_model("facial_emotion_detection_model.h5")
class_names=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

UPLOAD_FOLDER='static/uploads'
# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

def detect_face(image_path):
    """Detect if there's a face in the image using OpenCV"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read the image file"
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return False, "No face detected in the image"
        elif len(faces) > 1:
            return True, f"Multiple faces detected ({len(faces)} faces)"
        else:
            return True, "Face detected successfully"
            
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

def detect_emotion(image_path):
    img=image.load_img(image_path,target_size=(48,48),color_mode='grayscale')
    img_array=image.img_to_array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)
    prediction=model.predict(img_array)
    predicted_index=np.argmax(prediction)
    predicted_class=class_names[predicted_index]
    # Convert numpy float32 to Python float for JSON serialization
    confidence=float(round(prediction[0][predicted_index]*100,2))

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

def detect_emotion_from_array(img_array):
    """Detect emotion from numpy array (for live detection)"""
    # Resize and preprocess the image
    img_resized = cv2.resize(img_array, (48, 48))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    img_input = np.expand_dims(img_input, axis=-1)
    
    # Predict emotion
    prediction = model.predict(img_input, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    # Convert numpy float32 to Python float for JSON serialization
    confidence = float(round(prediction[0][predicted_index] * 100, 2))
    
    return predicted_class, confidence

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file uploaded. Please select an image file.")
        file=request.files['file']
        if file.filename=='':
            return render_template('upload.html', error="No file selected. Please choose an image file.")

        if file:
            # Generate a unique filename to prevent overwriting if multiple users upload same filename
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_path=os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # First, check if a face is detected
            face_detected, face_message = detect_face(file_path)
            
            if not face_detected:
                # Clean up the uploaded file if no face detected
                try:
                    os.remove(file_path)
                except:
                    pass
                return render_template('upload.html', 
                                     error=f"No face detected! {face_message}. Please upload an image with a clear, visible face.",
                                     no_face=True)
            
            # If face is detected, proceed with emotion detection
            try:
                emotion, confidence, plot_path = detect_emotion(file_path)
                
                # Make plot_path relative to static folder for Flask url_for
                display_plot_path = url_for('static', filename='uploads/' + os.path.basename(plot_path))
                display_image_path = url_for('static', filename='uploads/' + os.path.basename(file_path))

                return render_template('upload.html', 
                                     image_path=display_image_path, 
                                     emotion=emotion, 
                                     confidence=confidence, 
                                     plot_path=display_plot_path,
                                     face_message=face_message)
            except Exception as e:
                # Clean up files on error
                try:
                    os.remove(file_path)
                except:
                    pass
                return render_template('upload.html', error=f"Error processing image: {str(e)}")

    return render_template('upload.html')

@app.route("/live")
def live():
    return render_template('live.html')

@app.route("/process_frame", methods=['POST'])
def process_frame():
    """Process a single frame from the webcam for emotion detection"""
    try:
        # Get the base64 image data from the request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove the data:image/jpeg;base64, prefix
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'})
        
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion for this face
            emotion, confidence = detect_emotion_from_array(face_roi)
            
            results.append({
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'emotion': emotion,
                'confidence': float(confidence)  # Ensure it's a Python float
            })
        
        return jsonify({'faces': results})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/about")
def about():
    return render_template('about.html')

if __name__=="__main__":
    # Get the port from the environment variable (Render sets this)
    # Default to 5000 for local development if PORT env var is not set
    port = int(os.environ.get('PORT', 5600))
    # Bind to 0.0.0.0 to make it accessible from outside the container
    # Set debug=False for production deployments on Render
    app.run(debug=False, host='0.0.0.0', port=port)