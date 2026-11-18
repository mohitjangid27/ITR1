HEART DISEASE PREDICTOR + DIABETES RISK ASSESSMENT  (Anaconda guide)
=====================================================================

1) Unzip this folder.

2) Open "Anaconda Prompt" (Windows) or Terminal (Mac/Linux) and go to the folder:
       cd path\to\Heart_Disease_Predictor

3) Create the environment (one time):
       conda env create -f environment.yml
       conda activate heartapp
   (Or, in an existing environment:  pip install -r requirements.txt)

--- HEART DISEASE WEB APP ---------------------------------------------
4) Train the model (needs internet the first time to download the dataset):
       python train_model.py
   This creates model.joblib.

5) Launch the app:
       streamlit run app.py
   It opens at http://localhost:8501 - fill in the fields and click
   "Predict Heart Disease".

--- DIABETES PROJECT (terminal version) --------------------------------
       python diabetes_risk_assessment.py
   Trains models, saves charts to ./outputs, then asks for patient data.

Troubleshooting
- "Model not found": run step 4 first, in the same folder as app.py.
- "streamlit not recognized": make sure the environment is activated.
- Offline: the scripts fall back to synthetic data (for demo only).

Educational project only - not a medical diagnosis.
