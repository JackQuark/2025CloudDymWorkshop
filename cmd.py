# auto push

import os
import subprocess
import datetime

os.system("copy \Campbellsci\LoggerNet\CR1000_2_Data1min.dat .")

subprocess.call(["git", "add", "."])
subprocess.call(["git", "commit", "-m", "commit at " + str(datetime.datetime.now())])
subprocess.call(["git", "push", "-u", "origin", "cr1000"])
