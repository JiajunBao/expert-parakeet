# EECS476-Final-Project


Reproduce results:
```
# set up environment
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt

# train ALBERT
bash src/run_driver.sh
# test
python3 infer.py

# train hattn
bash src/run_training.sh
# test
python3 src/eval.py

# deploy the online webapp for inference (with our hattn model)
python3 src/run_hattn_viewer.py
```