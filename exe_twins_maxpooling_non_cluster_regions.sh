
#!/bin/bash

# defining a list of the region numbers to loop through

python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'SVLB' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-0' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'JPN-1' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'ALSK' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'HUD-0' --version 'twins_alt_v5_dbht' 
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'non_cluster_regions' --region 'HUD-1' --version 'twins_alt_v5_dbht' --oversampling 'True'

python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'non_cluster_regions' --region 'SVLB' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'non_cluster_regions' --region 'JPN-0' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'non_cluster_regions' --region 'JPN-1' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'non_cluster_regions' --region 'ALSK' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'non_cluster_regions' --region 'HUD-0' --version 'twins_alt_v5' 
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'non_cluster_regions' --region 'HUD-1' --version 'twins_alt_v5' --oversampling 'True'
