import React from "react";
import Webcam from 'react-webcam';
import { useState,useRef,useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import './Marking.css'
import Navbar from "../Home/Navbar";

function Marking_Attendance(){
    const webcamRef = useRef(null);
    const [message, setMessage] = useState('');
    const [model,setModel] = useState('facenet')

    useEffect(() => {
        // Cleanup function to stop webcam stream
        return () => {
            if (webcamRef.current) {
                const videoElement = webcamRef.current.video; // Get the video element from the webcamRef
                if (videoElement && videoElement.srcObject) {
                    const stream = videoElement.srcObject;
                    const tracks = stream.getTracks();
                    tracks.forEach(track => track.stop());
                }
            }
        };
    }, []);

    const handlemodel = (event) => {
        setModel(event.target.value);
    }
    

    const capture = async () => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
            const img = new Image();
            img.src = imageSrc;

            img.onload = async () => {
                const canvas = document.createElement('canvas');
                canvas.width = 160; // Desired width
                canvas.height = 160; // Desired height
                const ctx = canvas.getContext('2d');

                // Draw the image onto the canvas at the new size
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                const resizedImage = canvas.toDataURL('image/jpeg'); // Convert to base64
                // Send the captured image to the backend
            }
            try {
                const response = await axios.post('http://localhost:5000/recognize', {
                    image: imageSrc.split(',')[1], // Get base64 image data
                    model:model,
                },{
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });

                // Check the response from the backend
                if (response.data.success) {
                    alert(`${response.data.name} having Roll no : ${response.data.roll.toUpperCase()} has been Marked for Attendance!`)
                    setMessage(`${response.data.name} having Roll no : ${response.data.roll.toUpperCase()} has been Marked for Attendance!`);
                } 
                else if(!response.data.success){
                   alert(`${response.data.message}`);
                   setMessage(response.data.message);
                }
                else {
                    alert(`${response.data.message}`)
                    setMessage(response.data.message);
                }
            } catch (error) {
                console.error('Error comparing image:', error);
                setMessage('An error occurred while marking attendance.');
            }
        }
    };

    return (
        <>
        <Navbar/>
        <div className="container marker">
            <h1>Mark Attendance</h1>
            <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={400}
                height={300}
                className="web"
            />
            <select className="choice" name="choice" value={model} onChange={handlemodel}>
                <option value="facenet">Pretrained-Model</option>
                <option value="custom">Custom-CNN</option>
            </select>
            <button onClick={capture} className="capture-button">Capture</button>
        </div>
        </>
    );
}

export default Marking_Attendance;