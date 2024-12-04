import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import UserModel from "./models/students.js";

const app = express();
app.use(cors());
app.use(express.json());

mongoose
    .connect("mongodb://127.0.0.1:27017/face_recognition_db")
    .then(() => console.log("Connected to MongoDB"))
    .catch(err => console.error("Failed to connect to MongoDB", err));


    app.get("/getstudents", (req, res) => {
        const startOfDay = new Date();
        startOfDay.setUTCHours(0, 0, 0, 0); // UTC start
        const endOfDay = new Date();
        endOfDay.setUTCHours(23, 59, 59, 999); // UTC end
        
    
        UserModel.find({
            attendanceDate: {
                $gte: startOfDay, 
                $lte: endOfDay,  
            }
        })
            .then(users => {
                console.log("Today's Records:", users);
                res.json(users); 
            })
            .catch(err => {
                console.error("Query Error:", err);
                res.status(500).json({ error: err.message });
            });
    });
    


app.listen(3001,() => {
    console.log("Server is running on port 3001");
})
