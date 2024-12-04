import React, { useState, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './Register.css';
import Navbar from '../Home/Navbar';

function Register() {
    const [name, setName] = useState('');
    const [roll, setRoll] = useState('');
    const [photo, setPhoto] = useState(null);
    const navigate = useNavigate();

    const handleCapture = async (e) => {
        e.preventDefault();
        if (!name || !roll) {
            alert('Please fill all fields to capture a photo');
            return;
        }
        try {
            const response = await axios.post('http://localhost:5000/api/check_roll', { name, roll });

             
            if (!response.data.exists) {
                navigate(`/Webcam_face?name=${encodeURIComponent(name)}&roll=${encodeURIComponent(roll)}`);
            } else {
                alert(response.data.message); 
            }
        } catch (error) {
            if (error.response && error.response.status === 409) {
                alert(error.response.data.message);
            } else {
                console.error('An error occurred:', error);
                alert('Failed to check roll number. Please try again later.');
            }
        }
    };



    return (
        <>
        <Navbar/>
        <div className="container">
            <div className="register-box">
                <form>
                    <h1 className="head">Attendance Registration</h1>
                    <div className="input-box">
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Enter your name"
                        />
                    </div>
                    <div className="input-box">
                        <input
                            type="text"
                            value={roll}
                            onChange={(e) => setRoll(e.target.value)}
                            placeholder="Roll Number"
                        />
                    </div>
                    <button type="button" onClick={handleCapture} className="face">Capture Face</button>
                          <div className="marked">
              <p className='mark'>Already registered? <a onClick={() => {navigate('/Marking_Attendance')}}>Click here to mark Attendance</a></p>
             </div>
                </form>
            </div>
        </div>
        </>
    );
}

export default Register;
