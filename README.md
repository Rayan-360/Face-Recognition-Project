
# Face Recognition System

### A robust face recognition system for single-face scanning, multi-face scanning, and crowd analysis, built with **Flask** (backend) and integrated with **PyTorch** models. The system uses a **MongoDB database** to store embeddings and attendance records.

---

## 🌟 Features
- **Single Face Recognition**: Identify individuals in close-up images.
- **Group Face Recognition**: Detect and recognize multiple faces in group photos.
- **Crowd Analysis**: Perform large-scale face detection and recognition.
- **Attendance Tracking**: Automatically mark attendance based on recognized faces.
- **Seamless Backend Integration**: Flask-based REST API with PyTorch models.
- **Efficient Database**: Stores embeddings and attendance in MongoDB.

---

## 🚀 Technologies Used
- **Frontend**: React.js (for the user interface)
- **Backend**: Flask (Python), Tensorflow
- **Database**: MongoDB
- **Face Recognition Models**: 
  - MTCNN for face detection
  - InceptionResnetV1 for embedding generation
- **Other Tools**: 
  - Node.js for server management
  - RESTful API for client-server communication

---

## 🛠️ Project Setup

### Prerequisites
- Python 3.8+
- Node.js and npm
- MongoDB (local or cloud-based)
- Gaming laptop or a system with a GPU (optional for high-speed processing)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face-recognition-system
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

### Running the Project
1. Start the Flask backend servers:
   ```bash
   python main_server.py
   ```

2. Start the React frontend:
   ```bash
   cd frontend
   npm start
   ```

3. Open the app in your browser at:
   ```
   http://localhost:3000
   ```

---

## 🌀 API Endpoints

### **/recognize-single**
- **Method**: POST  
- **Description**: Identifies a single face from an image.
- **Payload**: `{ "image": "<base64_encoded_image>" }`
- **Response**: `{ "name": "<recognized_name>", "confidence": <value> }`

### **/recognize-group**
- **Method**: POST  
- **Description**: Detects and recognizes multiple faces from an image.
- **Payload**: `{ "image": "<base64_encoded_image>" }`
- **Response**: List of recognized faces.

### **/add-face**
- **Method**: POST  
- **Description**: Adds a new face embedding to the database.
- **Payload**: `{ "name": "<name>", "image": "<base64_encoded_image>" }`

---

## 🎯 Future Enhancements
- Optimize face detection for low-resolution images.
- Add real-time face recognition via webcam.
- Implement user authentication for secure access.
- Integrate additional analytics for crowd behavior insights.

---

## 💡 Project Insights
- Embedding Generation: Face embeddings are generated using **InceptionResnetV1** and stored in MongoDB.
- Real-time Processing: Efficient use of GPU for faster computations.
- Flexibility: Modular codebase to support future features.

---

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with detailed explanations.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact
For any queries or support, reach out to:
- **Developer**: Rayan  
- **Email**: [abd17rayan@gmail.com]  

