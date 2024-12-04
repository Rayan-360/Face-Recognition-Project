from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from pymongo import MongoClient
from PIL import Image
import io
import base64
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from tensorflow.keras import layers, models
import threading
import time

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models
mtcnn = MTCNN(keep_all=True, select_largest=False)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Load YOLO model for crowd analysis
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
collection = db['Registerations']
attendance_collection = db['Attendance_Validate']

# Global variables for webcam streaming
is_capturing = False
cap = None
capture_thread = None

#custom model

def create_model(input_shape=(96, 96, 3), embedding_dim=64):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(embedding_dim)
    ])
    return model

def extract_features(model, image):
    image = cv2.resize(image, (96, 96))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, 0)  # Add batch dimension
    features = model.predict(image)
    return features[0].numpy()  # Return the feature vector as a numpy array

def recognize_face_custom(custom_embedding, stored_custom_embeddings, stored_names, stored_rolls, threshold=0.7):
    best_match = "Unknown"
    best_roll = "Unknown"  

    # Loop through the stored embeddings in MongoDB
    for stored_custom_embedding, name, roll in zip(stored_custom_embeddings, stored_names, stored_rolls):
        # Compare the embeddings using cosine similarity
        similarity = cosine_similarity([stored_custom_embedding], [custom_embedding])[0][0]

        # If similarity is above the threshold, update the best match
        if similarity > threshold:
            best_match = name
            best_roll = roll  # Store the corresponding roll number
    # Return the name and roll if a match is found, otherwise 'Unknown'
    return best_match, best_roll,similarity

custom_model = create_model()  # Create the model
custom_model.compile(optimizer='adam', loss='triplet_loss')  # Compile the model




# Helper Functions
def preprocess_image(image):
    image = Image.fromarray(image).convert('RGB')
    image = image.resize((160, 160))
    image_np = np.array(image)
    face_tensor = torch.tensor(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return face_tensor


def load_embeddings_from_db():
    facenet_embeddings, custom_embeddings, names, rolls = [], [], [], []
    for document in collection.find():
        facenet_embeddings.append(np.array(document['facenet_embedding']))
        custom_embeddings.append(np.array(document['custom_cnn_embedding']))
        names.append(document['name'])
        rolls.append(document['roll'])
    return facenet_embeddings, custom_embeddings, names, rolls


# Function to convert an image file to tensor and generate embeddings
def image_to_tensor_embedding(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((160, 160))  # Resize for FaceNet
    image_np = np.array(image)
    face_tensor = torch.tensor(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # Normalize
    embedding = facenet_model(face_tensor).detach().numpy().flatten().tolist()
    return embedding


def recognize_face_facenet(embedding, stored_embeddings, stored_names, stored_rolls, threshold=0.85):
    min_dist = float('inf')
    name = 'Unknown'
    roll = 'Unknown'
    for idx, stored_embedding in enumerate(stored_embeddings):
        dist = np.linalg.norm(embedding - stored_embedding)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            name = stored_names[idx]
            roll = stored_rolls[idx]
    return name, roll

def recognize_faces(embedding, stored_embeddings, stored_names, stored_rolls, cosine_threshold=0.4, euclidean_threshold=0.7):
    recognized = []
    for idx, stored_embedding in enumerate(stored_embeddings):
        cosine_sim = cosine_similarity([embedding], [stored_embedding])[0][0]
        euclidean_dist = np.linalg.norm(embedding - stored_embedding)
        if cosine_sim > cosine_threshold and euclidean_dist < euclidean_threshold:
            recognized.append({
                'name': stored_names[idx],
                'roll': stored_rolls[idx],
                'similarity': float(cosine_sim),
                'distance': float(euclidean_dist)
            })
    return recognized

def preprocess_face(img, box, landmarks):
    face_img = img.crop(box).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
    face_tensor = (face_tensor - 0.5) / 0.5
    return face_tensor.unsqueeze(0)



def process_frame(frame):
    results = yolo_model(frame)
    persons = results.pandas().xyxy[0]
    persons = persons[persons['name'] == 'person']
    count = len(persons)
    for _, row in persons.iterrows():
        # cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
        cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 0, 0), 2)
    # cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, count

def capture_frames():
    global is_capturing, cap
    while is_capturing:
        ret, frame = cap.read()
        if ret:
            frame, count = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame_update', {'frame': frame_bytes, 'count': count})
        time.sleep(1)

# API Routes



@app.route('/api/generate-embedding', methods=['POST'])
def generate_embedding():
    try:
        # Parse input data
        data = request.get_json()
        name = data['name']
        roll = data['roll']
        image_data = base64.b64decode(data['image'].split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Generate FaceNet embedding
        face_tensor = preprocess_image(np.array(image))
        facenet_embedding = facenet_model(face_tensor).detach().numpy().flatten().tolist()

        
        # Preprocess image for custom model
        image_np = np.array(image.resize((96, 96))) / 255.0  # Normalize
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
        
        # Generate Custom CNN embedding
        custom_embedding = custom_model.predict(image_np)[0].tolist()

        # Store in MongoDB
        collection.insert_one({
            'name': name,
            'roll': roll.lower(),
            'facenet_embedding': facenet_embedding,
            'custom_cnn_embedding': custom_embedding
        })

        return jsonify({"message": "Embeddings stored successfully!"}), 201

    except Exception as e:
        # Log the error for debugging
        print(f"Error in generate_embedding: {e}")
        return jsonify({"error": str(e)}), 500



# New endpoint to compare captured image with stored embeddings
@app.route('/api/compare-embedding', methods=['POST'])
def compare_embedding():
    try:
        data = request.get_json()

        # Debugging: Print incoming data

        # Check if 'image' is in data and it is not empty
        if 'image' not in data or not data['image']:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 image
        image_data = base64.b64decode(data['image'].split(",")[1]) if ',' in data['image'] else None

        if image_data is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Generate embedding for the captured image
        embedding = image_to_tensor_embedding(image_data)

        # Fetch all embeddings from MongoDB and compare
        records = collection.find()
        for record in records:
            stored_embedding = np.array(record['facenet_embedding'])
            # Compare embeddings (using a threshold)
            if np.linalg.norm(stored_embedding - embedding) < 0.8:  # Adjust threshold as needed
                return jsonify({"success": True})

        return jsonify({"success": False})

    except Exception as e:
        print("Error processing request:", e)
        return jsonify({"error": "An error occurred during comparison"}), 500

@app.route('/api/check_roll',methods=['POST'])
def check_roll():
    data = request.json
    roll = data.get('roll')

    # Check if roll number already exists (case-insensitive)
    existing_user = collection.find_one({"roll": roll.lower()})
    
    if existing_user:
        return jsonify({"exists": True, "message": "Roll number is already registered"}), 409
    else:
        return jsonify({"exists": False, "message": "Roll number is available"}), 200

@app.route('/recognize', methods=['OPTIONS'])
def handle_options():
    return '', 200  # Respond with a status of 200 (OK)

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = base64.b64decode(data['image'])
    model_choice = data.get('model', 'facenet')

    img = Image.open(io.BytesIO(image_data))

    stored_embeddings,stored_custom_embeddings, stored_names, stored_rolls = load_embeddings_from_db()
    
    if model_choice=='facenet':
        face_tensor = preprocess_image(np.array(img))
        face_embedding = facenet_model(face_tensor).detach().numpy().flatten().tolist()
        name, roll = recognize_face_facenet(face_embedding, stored_embeddings, stored_names, stored_rolls)
    elif model_choice=='custom':
        image_np = np.array(img.resize((96, 96))) / 255.0  # Normalize
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
        # Generate Custom CNN embedding
        custom_embedding = custom_model.predict(image_np)[0].tolist()
        name,roll,similarity = recognize_face_custom(custom_embedding,stored_custom_embeddings,stored_names,stored_rolls)
        print(similarity)
    roll = roll.upper()

    if name == 'Unknown':
        return jsonify({'success': False, 'message': "User not registered!"})
    today_date = datetime.now().date()
    existing_record = attendance_collection.find_one({"roll": roll, "attendanceDate": {"$gte": datetime(today_date.year, today_date.month, today_date.day)}})
    if existing_record:
        return jsonify({'success': False, 'message': f"Attendance already marked today for Roll no : {roll}"})
    attendance_collection.insert_one({'name': name, 'roll': roll, 'attendanceDate': datetime.now()})
    return jsonify({'success': True, 'message': "Attendance marked successfully.", 'name': name, 'roll': roll})

@app.route('/recognize-group', methods=['POST'])
def recognize_group():
    data = request.json
    image_data = base64.b64decode(data.get('image'))
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return jsonify({'success': False, 'message': "No faces detected!"})
    stored_embeddings,stored_custom_embeddings, stored_names, stored_rolls = load_embeddings_from_db()
    results = []
    for i, box in enumerate(boxes):
        if probs[i] < 0.85:
            continue
        face_tensor = preprocess_face(img, box, landmarks[i])
        face_embedding = facenet_model(face_tensor).detach().numpy().flatten()
        recognized_faces = recognize_faces(face_embedding, stored_embeddings, stored_names, stored_rolls)
        if recognized_faces:
            results.extend(recognized_faces)
        else:
            results.append({'name': "Unknown", 'roll': "Unknown"})
    return jsonify({'success': True, 'results': results})


@socketio.on('start_stream')
def start_stream():
    global is_capturing, cap, capture_thread
    if not is_capturing:
        cap = cv2.VideoCapture(0)
        is_capturing = True
        capture_thread = threading.Thread(target=capture_frames)
        capture_thread.start()

@socketio.on('stop_stream')
def stop_stream():
    global is_capturing, cap
    is_capturing = False
    if cap:
        cap.release()

# if __name__ == '__main__':
#     socketio.run(app, debug=True, port=5000)


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)





















