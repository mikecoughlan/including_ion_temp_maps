
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-0'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-1'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-2'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-3'
