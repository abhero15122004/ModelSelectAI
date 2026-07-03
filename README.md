ModelSelectAI 🤖
ModelSelectAI is an Automated Machine Learning (AutoML) platform designed to easily train, evaluate, and select the best machine learning models for your datasets. It supports a variety of tasks including Tabular, Time Series, and Computer Vision modeling.

Tech Stack
Frontend: React.js, TailwindCSS (or Vanilla CSS)
Backend: Python, Flask
Machine Learning Engine: Scikit-learn, CatBoost, PyTorch (depending on task)

Project Structure

ModelSelectAI/
│
├── app.py                 # Main Flask server entry point
├── main.py                # Core ML pipeline execution
├── modelselectai/         # Python package containing ML logic and preprocessing
│   ├── models_tabular.py
│   ├── models_vision.py
│   ├── models_timeseries.py
│   └── ...
│
└── automl-frontend/       # React Frontend application
    ├── public/
    └── src/
        ├── components/
        └── services/api.js # API connection to the Flask backend
🛠️ Local Development Setup
Because this project is configured for local execution, you need to run both the frontend and the backend servers simultaneously.

1. Start the Backend (Flask)
Open a terminal in the root directory (ModelSelectAI) and run:

bash

# 1. Activate your virtual environment (if you use one)
.\.venv\Scripts\activate
# 2. Install dependencies
pip install -r requirements.txt
# 3. Start the Flask server
python app.py
The backend API will run at http://localhost:5050/api

2. Start the Frontend (React)
Open a second terminal, navigate into the automl-frontend folder, and run:

bash

# 1. Navigate to the frontend directory
cd automl-frontend
# 2. Install Node dependencies (only needed once)
npm install
# 3. Start the React development server
npm start
The React application will automatically open in your browser at http://localhost:3000

Environment Variables
Environment configurations are intentionally ignored in Git to prevent secrets from leaking. If setting this project up on a new machine, refer to automl-frontend/.env.example to see which variables are required. For local development, the frontend expects: REACT_APP_API_URL = http://localhost:5050/api

☁️ Deployment
This project was originally architected for cloud deployment (e.g., Azure for the backend, Netlify for the frontend). Ensure you update your CORS settings in app.py and your environment variables when pushing to a production environment.
