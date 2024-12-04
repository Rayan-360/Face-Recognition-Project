import React from "react";
import Navbar from "../Home/Navbar";
import './about.css'

function About(){

    return(
        <>
        <Navbar/>
        <div className="section">
            <div className="about-container">
                <div className="content-section">
                    <div className="title">
                           <h1>About Us</h1>
                    </div>
                    <div className="content">
                        <h3>Multi-Level Face Recognition and Crowd Analysis System</h3>
                         <p>We are dedicated to developing cutting-edge face recognition and crowd management solutions. 
                            Our goal is to enhance security, improve event management, and provide valuable insights 
                            through advanced AI and deep learning technologies.
                            </p>
                        <div className="read-btn">
                            <button>Read More</button>
                            <button>Start Now</button>
                        </div>
                    </div>
                </div>
                <div className="image-section">
                    <img src="./about.png"/>
                </div>
            </div>
        </div>
        </>
    );

}

export default About;