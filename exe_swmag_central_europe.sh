
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-0'
python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-1'
python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-2'
python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-3'


