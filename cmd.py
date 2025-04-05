# auto push

import subprocess
import datetime

subprocess.call(["git", "add", "."])
subprocess.call(["git", "commit", "-m", "commit at " + str(datetime.datetime.now())])
subprocess.call(["git", "push", "-u", "origin", "cr1000"])
