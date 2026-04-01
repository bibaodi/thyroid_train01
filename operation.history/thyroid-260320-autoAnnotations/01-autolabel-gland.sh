python /home/eton/00-srcs/ml_thyroid.01_etonTrainCodes/srcs/autoAnnotate/annotateByMLModel.py \
	--model_type segmentation \
	--model_file /mnt/datas/42workspace/34-project_ML_data_models_UltrasoundIntelligence/44-models/3-thyroid/model_segmentThyGland_v02.250821/model_segmentThyGland_v02.250821.pt \
	--label_name thyGland \
	--input_folder  /mnt/datas/42workspace/34-project_ML_data_models_UltrasoundIntelligence/10-datas4ML/3-thyroid/001-thyroidTestSetV01/hh_noduleOnly-21.dcm_frms/ \
	--output_folder  ./o3-test
