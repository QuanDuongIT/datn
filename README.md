Project Setup Instructions
Follow the steps below to set up and run the project coqui-ai/TTS vits:

1. Configure Model Path
Before running the application, make sure to set the model path and config path correctly in your application. You can configure the paths in the relevant part of your code:
# Configuring model and configuration file paths in app.py

model_path = "model/best_model.pth"
config_path = "model/config.json"

2. Create a Virtual Environment
To create an isolated Python environment, run the following command:
python -m venv venv

3. Activate the Virtual Environment
Once the virtual environment is created, activate it using the following command:
.\venv\Scripts\activate

4. Install Required Libraries
Install the necessary Python dependencies by running:
pip install -r requirements.txt

5. Run the Application
To start the application, use the following command:
python app.py

The app will be available at:
http://127.0.0.1:5000/

[32, 300, 400, 500, 600, 700, 800, 900, 1000],