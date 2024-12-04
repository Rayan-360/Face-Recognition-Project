import React from "react";
import './homepage.css'
import { Link } from "react-router-dom";
import { useState } from "react";

function Navbar(){

    const [sidebarOpen,setSidebarOpen] = useState(false);

    function toggle(){
        setSidebarOpen(!sidebarOpen);
    }

    const handleResize = () => {
        if (window.innerWidth > 680) {
            setSidebarOpen(false); 
        }
    };

    window.onresize = handleResize;



    return(
        <header className="header">
            <Link to="/" className="logo">IDentify</Link>
            <nav className={`sidebar ${sidebarOpen ? "open" : ""}`}>
            <Link onClick={toggle} className="arrow"><svg xmlns="http://www.w3.org/2000/svg" height="30px" viewBox="0 -960 960 960" width="30px" fill="#e8eaed"><path d="M647-440H160v-80h487L423-744l57-56 320 320-320 320-57-56 224-224Z"/></svg></Link>
            <Link to="/">Home</Link>
            <Link to="/records">Records</Link>
            <Link to="/register">Register</Link>
            <Link to="/about">About us</Link>
            </nav>
            <nav className="main-nav">
            <div className="rem">
                <Link to="/">Home</Link>
                <Link to="/records">Records</Link>
                <Link to="/register">Register</Link>
                <Link to="/about">About us</Link>
            </div> 
            <Link className="logo-side" onClick={toggle}><svg xmlns="http://www.w3.org/2000/svg" height="30px" viewBox="0 -960 960 960" width="30px" fill="#e8eaed"><path d="M120-240v-80h720v80H120Zm0-200v-80h720v80H120Zm0-200v-80h720v80H120Z"/></svg></Link>
            </nav>
        </header>
    );

}

export default Navbar;