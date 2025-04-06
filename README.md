python -m venv venv

.\venv\Scripts\activate


pip install -r requirements.txt


uvicorn main:app --reload

http://127.0.0.1:5000/
