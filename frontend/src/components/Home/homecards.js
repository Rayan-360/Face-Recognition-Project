import React from "react";
import Navbar from "./Navbar";
import './homepage.css'
import { Link } from "react-router-dom";


function Homecard(){


    return (
        <>
        <Navbar/>

        <h1 className="mode">CHOOSE A RECOGNITION MODE</h1>
        <div className="mode-container">
        <div className="card">
           <h2>Single Face Scanning</h2>
           <img src="./single.jpg"/>
           <p>Identify a single face in real-time for quick, precise recognition.</p>
           <button><Link to="/Marking_Attendance">Identify Face</Link></button>
        </div>
        <div className="card">
           <h2>Multi Face Scanning</h2>
           <img src="./multi.png"/>
           <p>Easily recognize multiple faces in one frame for faster processing.</p>
           <button><Link to="/multiscan">Start Group Scan</Link></button>
        </div>
        <div className="card">
           <h2>Crowd Detection</h2>
           <img src="./crowd.png"/>
           <p>Analyze large crowds for comprehensive face detection and head count.</p>
           <button><Link to="/crowd">Start Crowd Scan</Link></button>
        </div>
        </div>
        </>
    );


}

export default Homecard;