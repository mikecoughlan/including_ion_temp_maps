
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'SVLB'
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-0'
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-1'
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'ALSK'
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'HUD-0'
python3 swmag_model.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'HUD-1'


