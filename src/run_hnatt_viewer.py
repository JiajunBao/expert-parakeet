import os
import subprocess

os.environ['FLASK_APP'] = "src/app/app.py"
os.environ['FLASK_DEBUG'] = "1"
subprocess.call(['flask', 'run'])