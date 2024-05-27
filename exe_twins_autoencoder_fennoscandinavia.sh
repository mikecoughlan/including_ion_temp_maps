
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one

bash exe_twins_maxpooling_fennoscandinavia.sh

python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-0'
python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-1'
python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-2'
python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-3'
python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-4'
python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-5'
python3 twins_model_v_autoencoder.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-6'

bash exe_twins_autoencoder_greenland.sh