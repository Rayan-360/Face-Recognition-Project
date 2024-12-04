import ReactDOM from "react-dom/client";
import React from "react";
import {BrowserRouter,Routes,Route } from "react-router-dom";
import Register from "./components/single_face/Register";
import Webcam_face from "./components/single_face/Webcam_face"
import Marking_Attendance from "./components/single_face/Marking_Attendance";
import Homecard from "./components/Home/homecards";
// import Navbar from "./components/Home/Navbar";
import MultiFaceRecognition from "./components/multi_face/Multi_face_recognition";
import Crowd from "./components/crowd_analysis/crowdcount";
import About from "./components/others/about";
import Dashboard from "./components/Home/dashboard";
import Records from "./components/records/attendance_records";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Homecard/>}/>
        <Route path="/records" element = {<Records/>}/>
        <Route path="/register" element={<Register/>}/>
        <Route path="/Webcam_face" element={<Webcam_face/>}/>
        <Route path="/Marking_Attendance" element={<Marking_Attendance/>}/>
        <Route path="/multiscan" element={<MultiFaceRecognition/>}/>
        <Route path="/crowd" element={<Crowd/>}/>
        <Route path="/about" element={<About/>}/>
      </Routes>
    </BrowserRouter>

  );
}
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
export default App;
