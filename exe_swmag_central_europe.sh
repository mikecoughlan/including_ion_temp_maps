
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-0' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-0' --version 'extended_v0-1' 

python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-1' --version 'extended_v0-1_dbht'
python3 swmag_model.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-1' --version 'extended_v0-1' 

python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-2' --version 'extended_v0-1_dbht'
python3 swmag_model.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-2' --version 'extended_v0-1'

python3 swmag_model.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-3' --version 'extended_v0-1_dbht'
python3 swmag_model.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-3' --version 'extended_v0-1' 



