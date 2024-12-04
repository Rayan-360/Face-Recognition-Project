import React from "react";
import Navbar from "./Navbar";
import './dashboard.css';

function Dashboard(){

    return (
        <div className="dashboard">
        <Navbar/>
        <div className="dash-container">
        <h1 className="dash-heading">FACE <span style={{color:"#e81659"}}>RECOGNITION</span></h1>
        <p className="dash-content">Welcome to IDentify, your gateway to cutting-edge face recognition technology.
             Powered by advanced AI, we deliver seamless identity verification for individuals, groups, and crowds.
              With a focus on accuracy, speed, and security, IDentify is transforming recognition systems for a smarter, 
              connected future.
        </p>
        <div className="dash-button">
            <button style={{backgroundColor:"#e81659"}}>START NOW</button>
        </div>
        </div>
        </div>
    );
}

export default Dashboard;