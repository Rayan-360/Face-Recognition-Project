import React, { useState, useEffect } from "react";
import Navbar from "../Home/Navbar";
import "./records.css";
import axios from "axios";

function Records() {
    const [record, setRecord] = useState([]);

    useEffect(() => {
        axios
            .get("http://localhost:3001/getstudents")
            .then(users => setRecord(users.data))
            .catch(err => console.log(err));
    }, []);

    const formatDate = dateStr => {
      const date = new Date(dateStr);
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0"); // Months are 0-indexed
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}/${month}/${day}`;
  };

    return (
        <>
            <Navbar />
            <div className="table-container">
                <table className="table-records" border={1}>
                    <thead>
                        <tr>
                            <th>SNO</th>
                            <th>Name</th>
                            <th>Roll number</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        {record.length > 0 ? (
                            record.map((student, index) => (
                                <tr key={index}>
                                    <td>{index + 1}</td>
                                    <td>{student.name.toUpperCase()}</td>
                                    <td>{student.roll.toUpperCase()}</td>
                                    <td>
                                        {formatDate(student.attendanceDate)}
                                    </td>
                                </tr>
                            ))
                        ) : (
                            <tr>
                                <td colSpan="4">No records found</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </>
    );
}

export default Records;
