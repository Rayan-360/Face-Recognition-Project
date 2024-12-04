import React, { useState, useEffect, useCallback } from 'react';
import io from 'socket.io-client';
import './crowd.css'
import Navbar from '../Home/Navbar';


const socket = io('http://localhost:5000');

export default function Crowd() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [frame, setFrame] = useState('');
  const [count, setCount] = useState(0);
  const [error, setError] = useState(null);

  const startStream = useCallback(() => {
    socket.emit('start_stream');
    setIsStreaming(true);
  }, []);

  const stopStream = useCallback(() => {
    socket.emit('stop_stream');
    setIsStreaming(false);
    setCount(0);
  }, []);

  useEffect(() => {
    socket.on('connect_error', (error) => {
      console.error('Connection Error:', error);
      setError('Failed to connect to the server. Please try again.');
    });

    socket.on('frame_update', (data) => {
      setFrame(`data:image/jpeg;base64,${data.frame}`);
      setCount(data.count);
    });

    return () => {
      socket.off('frame_update');
      socket.off('connect_error');
    };
  }, []);

  return (
    <>
    <Navbar/>
    <div className="crowd-analysis-container">
      <h1 className="crowd-analysis-title">Live Crowd Analysis</h1>
      <div className="crowd-analysis-content">
      <div className="people-count" >
          <h2>Head Count: {count}</h2>
        </div>
        <div className="video-container">
          {isStreaming ? (
            <img src={frame} alt="Video feed" className="video-feed" />
          ) : (
            <img src='./video.png' className='logo'/>
          )}
        </div>
        <div className="stream-controls">
          <button
            onClick={startStream}
            disabled={isStreaming}
            className="start"
          >
            Start Stream
          </button>
          <button
            onClick={stopStream}
            disabled={!isStreaming}
            className="stop"
          >
            Stop Stream
          </button>
        </div>

        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}
      </div>
    </div>
    </>
  );
}
