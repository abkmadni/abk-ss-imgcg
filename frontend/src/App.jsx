import "./App.css";
import { useState, useRef, useEffect } from "react";

export default function ImageCaptionGenerator() {
  const [imageSrc, setImageSrc] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [caption, setCaption] = useState("");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Start camera with permission and secure context checks
  const startCamera = async () => {
    setCaption("");
    setImageSrc(null);

    if (!window.isSecureContext) {
      alert(
        "Camera access requires a secure context (HTTPS). Please serve this app over HTTPS."
      );
      return;
    }

    setCameraOn(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.error(err);
      if (err.name === "NotAllowedError") {
        alert(
          "Permission denied: please allow camera access in your browser settings."
        );
      } else if (err.name === "NotFoundError") {
        alert("No camera device found. Please connect a camera.");
      } else {
        alert(`Unable to access camera: ${err.message}`);
      }
      setCameraOn(false);
    }
  };

  // Stop camera safely
  const stopCamera = () => {
    const videoEl = videoRef.current;
    if (videoEl && videoEl.srcObject) {
      const stream = videoEl.srcObject;
      stream.getTracks().forEach((track) => track.stop());
      videoEl.srcObject = null;
    }
    setCameraOn(false);
  };

  // Capture photo from video
  const capturePhoto = () => {
    const videoEl = videoRef.current;
    if (!videoEl || !videoEl.videoWidth) {
      alert("Camera not initialized.");
      return;
    }
    const canvas = canvasRef.current;
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg");
    setImageSrc(dataUrl);
    stopCamera();
  };

  // Handle file upload
  const uploadImage = () => {
    setCaption("");
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setImageSrc(ev.target.result);
    reader.readAsDataURL(file);
  };

  // Send image for caption generation
  const generateCaption = async () => {
    if (!imageSrc) {
      alert("Please upload or capture an image first.");
      return;
    }
    try {
      const response = await fetch("/api/generate_caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageSrc }),
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }
      const data = await response.json();
      setCaption(data.caption || "No caption returned.");
    } catch (err) {
      console.error(err);
      alert(`Caption generation failed: ${err.message}`);
    }
  };

  useEffect(() => () => stopCamera(), []);

  return (
    <div className="min-h-screen bg-gray-50 p-6 md:p-12">
      <div className="max-w-3xl mx-auto bg-white shadow-md rounded-lg p-8">
        {/* Centered logo */}
        <div className="flex justify-center mb-6">
          <img src="logo.png" alt="Logo" className="w-15" />
        </div>

        <h1 className="text-4xl font-extrabold text-gray-800 mb-6 text-center">
          Image Caption Generator
        </h1>

        <div className="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-6 mb-6">
          <button
            onClick={uploadImage}
            className="w-full md:w-auto px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white font-semibold rounded-md shadow"
          >
            Upload Image
          </button>
          <button
            onClick={startCamera}
            className="w-full md:w-auto px-6 py-3 bg-green-500 hover:bg-green-600 text-white font-semibold rounded-md shadow"
          >
            Open Camera
          </button>
        </div>

        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />

        {cameraOn && (
          <div className="relative mb-6">
            <video
              ref={videoRef}
              className="mx-auto w-full max-w-md rounded-lg shadow-lg"
            />
            <button
              onClick={capturePhoto}
              className="absolute bottom-4 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-full shadow-lg"
            >
              Capture
            </button>
          </div>
        )}

        {imageSrc && (
          <div className="mb-6 text-center">
            <img
              src={imageSrc}
              alt="Selected"
              className="inline-block max-h-64 object-contain rounded-lg shadow-md"
            />
          </div>
        )}

        <div className="text-center mb-8">
          <button
            onClick={generateCaption}
            className="px-8 py-3 bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-lg shadow-lg"
          >
            Generate Caption
          </button>
        </div>

        {caption && (
          <div className="bg-gray-100 p-6 rounded-lg shadow-inner">
            <p className="text-center text-xl font-medium text-gray-800">
              {caption}
            </p>
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
