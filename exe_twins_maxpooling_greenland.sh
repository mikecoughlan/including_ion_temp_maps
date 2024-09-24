
#!/bin/bash

python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'greenland_cluster' --region 'GRL-0' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'greenland_cluster' --region 'GRL-1' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'greenland_cluster' --region 'GRL-2' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'greenland_cluster' --region 'GRL-3' --version 'twins_alt_v5_dbht' 

python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-0' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-1' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-2' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-3' --version 'twins_alt_v5' 
