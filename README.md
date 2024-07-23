# DP-Mondrian Clustering
The implementation of the differentially private clustering algorithm DPM in the paper 'DPM: Clustering Sensitive Data through Separation'.

## Setup
Install **Python>=3.10** and download the required packages:
```bash
pip install -r requirements.txt
```
## Usage
DPM can be used as follows:
```Python
from dpm import DPM

dpm = DPM(
    data=      # numpy.ndarray, shape = (num_points, num_dims) 
    bounds=    # tuple, (min(data), max(data)) (Potential privacy risk!)
    epsilon=   # float, epsilon > 0 
    delta=     # float, delta < 1
)

centres, clusters = dpm.perform_clustering()
# ^        ^
# |        |
# |        --------- list, len(clusters) = k, array indices of data for each cluster
# ------------------ np.ndarray, shape = (k, num_dims)
```
In `demo.py` you can find a full example of how DPM can be used to cluster a data set.

## Citation
If you use the code, please consider citing our Paper "DPM: Clustering Sensitive through Separation". \<conference\>.
```
Coming soon...
```
