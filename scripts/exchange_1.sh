python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:4" --dataset_name "ExchangeRate" --supervised "ft" --adapter "pca"
python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:4" --dataset_name "ExchangeRate" --supervised "ft" --adapter "pca"
python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:4" --dataset_name "ExchangeRate" --supervised "ft" --adapter "pca"
python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:4" --dataset_name "ExchangeRate" --supervised "ft"
python run.py --forecast_horizon 192 --model_name "AutonLab/MOMENT-1-small" --context_length 512 --seed $RANDOM --device "cuda:4" --dataset_name "ExchangeRate" --adapter "dropoutLinearAE" --use_revin --supervised "ft_then_supervised"
