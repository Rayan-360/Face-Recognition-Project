import React, { useState, useRef ,useEffect} from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Navbar from '../Home/Navbar';
import './multiface.css';

function MultiFaceRecognition() {
    const webcamRef = useRef(null);
    const [capturedImage, setCapturedImage] = useState(null);
    const [attendanceResults, setAttendanceResults] = useState([]);
    const [error, setError] = useState("");
    const [position,setPosition] = useState("");

    const captureImage = () => {
        const imageSrc = webcamRef.current.getScreenshot();
        setCapturedImage(imageSrc);
    };

    const markAttendance = async () => {
        if (!capturedImage) {
            setError("Please capture an image first.");
            return;
        }

        try {
            const response = await axios.post("http://localhost:5000/recognize-group", {
                image: capturedImage.split(",")[1]  // Send base64 data without prefix
            });

            if (response.data.success) {
                setAttendanceResults(response.data.results);
                setError("");
                console.log(attendanceResults);
            } else {
                setError("No faces detected!");
            }
        } catch (error) {
            console.error("Error recognizing faces:", error);
            setError("An error occurred while recognizing faces. Please try again.");
        }
    };

    useEffect(() => {
        return () => {
            if (webcamRef.current && webcamRef.current.video.srcObject) {
                let stream = webcamRef.current.video.srcObject;
                stream.getTracks().forEach(track => track.stop()); // Stop all video tracks
            }
        };
    }, []);

    const handleResize = () => {
        if (window.innerWidth <661){
            setPosition("bottom");
            
        }
        else{
            setPosition("");
        }
    }

    return (
        <>
            <Navbar />

            <h1 className="recognition-title">Multi-Face Recognition System</h1>
            <div className="recognition-container">

                
                <div className={`left ${attendanceResults.length > 0 ? 'has-results' : ''}`}>
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="webcam-display"
                    />
                    <div className="button-container">
                        <button onClick={captureImage} className="capture">
                            Capture Image
                        </button>
                        <button onClick={markAttendance} className="recognize">
                            Recognize Faces
                        </button>
                    </div>
                </div>

                <div className="right">
                    {attendanceResults.length > 0 && (
                        <div className="results-container">
                            <h2 className="results-title">HEAD COUNT : {attendanceResults.length}</h2>
                            <ul className="results-list">
                                {attendanceResults.map((result, index) => (
                                    <li key={index} className="results-item">
                                        {result.name !== "Unknown"
                                            ? `Person ${index + 1} : ${result.name} (ROLL NO: ${result.roll.toUpperCase()})`
                                            : "Unknown Face Detected"}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>

                {error && (
                    <div className="error-message">
                        <p>{error}</p>
                    </div>
                )}
            </div>
        </>
    );
}

export default MultiFaceRecognition;
