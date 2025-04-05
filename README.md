# WXT cr1000 dataset

### how to load
use `git clone` or directly download the file: *CR1000_2_Data1min.dat*

e.g.
```python
import pandas as pd
fpath = "./CR1000_2_Data1min.dat"
with open(fpath, 'r') as f:
    _ = f.readline()
    headers = f.readline().replace('"', '').split(',')
df = pd.read_csv(fpath, sep=',', names=headers, skiprows=15567)
```
