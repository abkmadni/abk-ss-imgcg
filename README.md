# Image Caption Generator

## Project Overview
The Image Caption Generator is a web application that allows users to upload or capture an image and receive a natural language caption describing the content of the image. It consists of two main components:
- **Backend (Flask + TensorFlow)**: Extracts image features using Xception and generates captions using a trained LSTM model saved as an H5 file (`models/best_model_9.h5`).
- **Frontend (React + Tailwind CSS)**: Provides a user-friendly interface for uploading/capturing images and displaying the generated captions.

## Folder Structure
```
project-root/
├── backend/
│   ├── generate_caption.py
│   ├── tokenizer.p
│   ├── requirements.txt
│   └── models/
│       └── best_model_9.h5
├── frontend/
│   ├── public/
│   │   └── logo.png
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   ├── main.jsx
│   │   └── index.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
├── LICENSE
└── README.md
```

## Prerequisites
- **Backend**:
  - Python 3.8 or higher
  - `pip` package manager
- **Frontend**:
  - Node.js 16.x or higher
  - `npm` (comes with Node.js)

## Backend Setup (Flask Server)

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a Python virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should contain:
   ```
   Flask
   tensorflow-cpu
   numpy
   Pillow
   h5py
   ```

4. **Verify model and tokenizer files**:
   - Ensure `tokenizer.p` is present directly in `backend/`.
   - Ensure the trained caption model `best_model_9.h5` exists at `backend/models/best_model_9.h5`.

5. **Run the Flask server**:
   ```bash
   python generate_caption.py
   ```
   By default, the server listens on `http://0.0.0.0:5000`. You should see console messages indicating that the tokenizer, caption model, and Xception model have loaded.

6. **Test the endpoint** (optional):
   Send a POST request to the endpoint using `curl` or Postman:
   ```bash
   curl -X POST http://127.0.0.1:5000/api/generate_caption      -H "Content-Type: application/json"      -d "{"image":"data:image/jpeg;base64,/9j/4AAQ..."}"
   ```
   Replace the base64 string with your own. The response should be a JSON object:
   ```json
   {
     "caption": "a dog is running on the grass"
   }
   ```

## Frontend Setup (React + Tailwind CSS)

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   This will start Vite on `http://localhost:5173`.

4. **Open your browser**:
   Navigate to `http://localhost:5173`. You should see the Image Caption Generator interface.

## Usage Flow

1. **Upload or Capture Image**:  
   - Click “Upload Image” to select a file from your computer.  
   - Or click “Open Camera” to capture a photo from your webcam (if supported).

2. **Display Preview**:  
   - The selected/captured image will display on the page within a styled container.

3. **Generate Caption**:  
   - Click the “Generate Caption” button.  
   - The frontend encodes the image as a base64 data URL and sends it via POST to `http://localhost:5000/api/generate_caption`.

4. **Receive and Show Caption**:  
   - The Flask backend decodes the image, extracts features with Xception, and uses the LSTM model (`best_model_9.h5`) to produce a caption.  
   - The response JSON contains `{ "caption": "..." }`, which the React app displays in a styled box below the button.

## File Descriptions

- **backend/generate_caption.py**:  
  Contains the Flask server code that:
  - Loads `tokenizer.p`, `Xception` model, and `best_model_9.h5`.  
  - Defines `/api/generate_caption` to accept a JSON with a data URL, extract features, run LSTM, and return a caption.

- **backend/tokenizer.p**:  
  The pickled tokenizer used to convert words to/from indices when generating captions.

- **backend/models/best_model_9.h5**:  
  The trained LSTM‐based captioning model.

- **frontend/src/App.jsx**:  
  - Defines the React component(s) for uploading/capturing an image.  
  - Manages state: `imageSrc`, `caption`, etc.  
  - Renders buttons styled with Tailwind classes.  
  - Sends the image to the Flask API and displays the returned caption.

## Building for Production

- **Backend**: The Flask server remains the same. Run `python generate_caption.py` on your production host (e.g., VPS or any server).  
- **Frontend**:
  1. From `frontend/` run:
     ```bash
     npm run build
     ```
  2. This creates a static `dist/` folder containing the optimized React + Tailwind app.  
  3. Serve the contents of `dist/` via any static‐file host (e.g., nginx, Apache, or a CDN).  
  4. Ensure the static build’s API calls still point to the Flask server’s production URL (e.g., `https://yourdomain.com/api/generate_caption`).

## Troubleshooting

- **Missing Packages**:  
  If you see `ModuleNotFoundError`, make sure you installed exactly the versions in `backend/requirements.txt` and `frontend/package.json`.

- **Model or Tokenizer Not Found**:  
  Verify paths:
  - `tokenizer.p` in `backend/`  
  - `models/best_model_9.h5` in `backend/models/`

- **Slow First Inference**:  
  The first run of Xception may take a few seconds as TensorFlow loads weights. Subsequent inferences will be faster.

- **CORS Errors During Development**:  
  Since the frontend runs on `localhost:3000` and backend on `localhost:5000`, you may need to install `flask-cors` and add:
  ```python
  from flask_cors import CORS
  CORS(app)
  ```
  at the top of `generate_caption.py`.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
