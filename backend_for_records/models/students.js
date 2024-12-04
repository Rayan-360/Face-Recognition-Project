import mongoose from "mongoose";

const UserSchema = new mongoose.Schema({
    name:String,
    roll:String,
    attendanceDate: {
        type: Date,
        get: (date) => date?.toISOString().split('T')[0], // Format to 'YYYY-MM-DD'
    },
},{ timestamps: true, toJSON: { getters: true } });

const UserModel = mongoose.model("Attendance_Validate", UserSchema, "Attendance_Validate");


export default UserModel;