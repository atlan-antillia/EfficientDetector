python model_inspect.py --runmode=saved_model_infer --model_name=efficientdet-d0 --saved_model_dir=./projects/BloodCells/saved_model --min_score_thresh=0.3 --hparams=./projects/BloodCells/configs/detect.yaml --input_image=./projects/BloodCells/test/*.jpg --output_image_dir=./projects/BloodCells/outputs