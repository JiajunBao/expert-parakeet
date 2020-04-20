# EECS476-Final-Project


Reproduce results:
```
# set up environment
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt

# train ALBERT
bash src/run_driver.sh

# train hattn

bash src/run_training.sh

```