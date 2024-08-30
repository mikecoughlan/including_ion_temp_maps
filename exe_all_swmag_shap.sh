
#!/bin/bash

# defining a list of the region numbers to loop through

# Loop through regions and run non_twins_modeling_final_version for each one


python3 calculating_shap.py --target 'rsd' --cluster 'canadian_cluster' --region 'CAN-0' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'canadian_cluster' --region 'CAN-1' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'canadian_cluster' --region 'CAN-2' --version 'swmag_alt_v4' --model_type 'swmag'

python3 calculating_shap.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-0' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-1' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-2' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'central_european_cluster' --region 'CEU-3' --version 'swmag_alt_v4' --model_type 'swmag'

python3 calculating_shap.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-0' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-1' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-2' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'greenland_cluster' --region 'GRL-3' --version 'swmag_alt_v4' --model_type 'swmag'

python3 calculating_shap.py --target 'rsd' --cluster 'non_cluster_regions' --region 'SVLB' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'non_cluster_regions' --region 'JPN-0' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'non_cluster_regions' --region 'JPN-1' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'non_cluster_regions' --region 'ALSK' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'non_cluster_regions' --region 'HUD-0' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'non_cluster_regions' --region 'HUD-1' --version 'swmag_alt_v4' --model_type 'swmag'

python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-0' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-1' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-2' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-3' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-4' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-5' --version 'swmag_alt_v4' --model_type 'swmag'
python3 calculating_shap.py --target 'rsd' --cluster 'fennoscandinavian_cluster' --region 'FSC-6' --version 'swmag_alt_v4' --model_type 'swmag'
