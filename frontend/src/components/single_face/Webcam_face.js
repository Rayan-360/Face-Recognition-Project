import React, { useRef, useEffect } from 'react';
import { useLocation,useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Webcam_face.css'
import Navbar from '../Home/Navbar';

function CaptureFace() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const location = useLocation();
    const navigate = useNavigate();

    // Extract name and mno from URL parameters
    const params = new URLSearchParams(location.search);
    const name = params.get('name');
    const roll = params.get('roll');

    useEffect(() => {
        const startVideo = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoRef.current.srcObject = stream;
            } catch (error) {
                console.error("Error accessing camera:", error);
            }
        };

        startVideo();

        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                const stream = videoRef.current.srcObject;
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop()); // Stop each track
                videoRef.current.srcObject = null; // Clear the video source
            }
        };
    }, []);

    const handleCaptureAndSend = () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        const context = canvas.getContext('2d');
        
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const offScreenCanvas = document.createElement('canvas');
        offScreenCanvas.width = 160;
        offScreenCanvas.height = 160;
        const offScreenContext = offScreenCanvas.getContext('2d');
        offScreenContext.drawImage(canvas, 0, 0, 160, 160);

        const imageSrc = offScreenCanvas.toDataURL('image/jpeg');
        
        const data = { name, roll, image: imageSrc };
        
        axios.post('http://localhost:5000/api/generate-embedding', data)
            .then((response) => {
                alert('Registration successful');
                // setTimeout(() => {
                //     navigate("/register")
                // },2000)
            })
            .catch((error) => {
                console.error("Error:", error);
                alert('Error in registration');
            });
    };

    return (
        <>
        <Navbar/>
        <div className='register-face'>
            <h1>Capture Face</h1>
            <video ref={videoRef} autoPlay className="cam" width="400" height="300"></video>
            <canvas className='canvas1' ref={canvasRef} width="400" height="300" style={{ display: "none" }}></canvas>
            <button onClick={handleCaptureAndSend} className='register-button'>Capture & Register</button>
        </div>
        </>
    );
}

export default CaptureFace;

