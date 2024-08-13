
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'greenland_cluster' --region 'GRL-0'
# python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'greenland_cluster' --region 'GRL-2'
# python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'SVLB'
# python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-0'
# python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-1'
