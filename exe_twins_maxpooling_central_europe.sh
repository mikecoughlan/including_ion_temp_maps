
#!/bin/bash

# defining a list of the region numbers to loop through

python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-0' --version 'twins_alt_v5_dbht' 
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-1' --version 'twins_alt_v5_dbht' 
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-2' --version 'twins_alt_v5_dbht' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'dbht_max' --cluster 'central_european_cluster' --region 'CEU-3' --version 'twins_alt_v5_dbht' 

python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-0' --version 'twins_alt_v5' 
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-1' --version 'twins_alt_v5' 
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-2' --version 'twins_alt_v5' --oversampling 'True'
python3 twins_model_v_maxpooling.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-3' --version 'twins_alt_v5' 

