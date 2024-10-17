
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'SVLB' --version 'extended_v0-1_dbht'
python3 swmag_model.py --target 'rsd' --cluster 'non_cluster_regions' --region 'SVLB' --version 'extended_v0-1'

python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-0' --version 'extended_v0-1_dbht'
python3 swmag_model.py --target 'rsd' --cluster 'non_cluster_regions' --region 'JPN-0' --version 'extended_v0-1'

python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-1' --version 'extended_v0-1_dbht'
python3 swmag_model.py --target 'rsd' --cluster 'non_cluster_regions' --region 'JPN-1' --version 'extended_v0-1'

python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'ALSK' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'non_cluster_regions' --region 'ALSK' --version 'extended_v0-1' 

python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'HUD-0' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'non_cluster_regions' --region 'HUD-0' --version 'extended_v0-1' 

python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'HUD-1' --version 'extended_v0-1_dbht' 
python3 swmag_model.py --target 'rsd' --cluster 'non_cluster_regions' --region 'HUD-1' --version 'extended_v0-1' 
