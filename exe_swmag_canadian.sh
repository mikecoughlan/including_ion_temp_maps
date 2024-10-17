
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 swmag_model.py --target 'dbht_max' --cluster 'canadian_cluster' --region 'CAN-0' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'canadian_cluster' --region 'CAN-0' --version 'extended_v0-1' 

python3 swmag_model.py --target 'dbht_max' --cluster 'canadian_cluster' --region 'CAN-1' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'canadian_cluster' --region 'CAN-1' --version 'extended_v0-1' 

python3 swmag_model.py --target 'dbht_max' --cluster 'canadian_cluster' --region 'CAN-2' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'canadian_cluster' --region 'CAN-2' --version 'extended_v0-1' 
