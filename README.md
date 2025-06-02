<!-- điều kiện triển khai trên vs code, python 3.9 -->
0. Training
Có thể train model bằng dataset khác theo file code training_vits.ipynb

1. Create a Virtual Environment
<!-- To create an isolated Python environment, run the following command: -->
python -m venv venv

2. Activate the Virtual Environment
<!-- Once the virtual environment is created, activate it using the following command: -->
.\venv\Scripts\activate
<!-- deactivate để ngắt -->

3. Dowload and Configure Model Path(có thể chung 1 config khi chỉ triển khai)
mkdir vits2_pytorch/models
cd vits2_pytorch/models
<!-- dowload css10 model checkpoit G_12000 link: https://drive.google.com/file/d/10LPmH50a-vidAPviSL7stkW6zTFlEcWY/view?usp=sharing
    dowload css10 config checkpoit G_12000 link: https://drive.google.com/file/d/1-Fq5cjaGY_OaBovaKvqkOZieFkapbUks/view?usp=sharing
    dowload jsut model checkpoit G_30000 link: https://drive.google.com/file/d/12H0e0Q9ovycNqpfmOUxOlZ5FK1Ly5vr3/view?usp=sharing
    dowload jsut config checkpoit G_30000 link: https://drive.google.com/file/d/116jnKove3n6EtUWNTTmG_RolQLQp1vWu/view?usp=sharing
 -->
pip install gdown
gdown --id 10LPmH50a-vidAPviSL7stkW6zTFlEcWY
gdown --id 1-Fq5cjaGY_OaBovaKvqkOZieFkapbUks
gdown --id 12H0e0Q9ovycNqpfmOUxOlZ5FK1Ly5vr3
gdown --id 116jnKove3n6EtUWNTTmG_RolQLQp1vWu

4. Install Required Libraries
<!-- Install the necessary Python dependencies by running: -->
<!-- cd vào folder /datn để cài requirements.txttxt -->
cd ../../        
pip install -r requirements.txt

5. Run the Application
To start the application, use the following command:
python app.py

The app will be available at:
http://127.0.0.1:5000/
