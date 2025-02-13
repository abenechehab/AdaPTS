python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:5" --dataset_name "Weather" --supervised "ft"
python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:5" --dataset_name "Weather" --supervised "ft" --adapter "pca"
python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:5" --dataset_name "Weather" --supervised "ft" --adapter "pca"
