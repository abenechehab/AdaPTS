---
license: mit
datasets:
- AutonLab/Timeseries-PILE
metrics:
- accuracy
- mse
- mae
- f1
tags:
- time series
- forecasting
- classification
- anomaly detection
- imputation
- transformers
- pretrained models
- foundation models
- time-series
pipeline_tag: time-series-forecasting
---
# MOMENT-Small

MOMENT is a family of foundation models for general-purpose time-series analysis. The models in this family (1) serve as a building block for diverse **time-series analysis tasks** (e.g., forecasting, classification, anomaly detection, and imputation, etc.), (2) are effective **out-of-the-box**, i.e., with no (or few) task-specific exemplars (enabling e.g., zero-shot forecasting, few-shot classification, etc.), and (3) are **tunable** using in-distribution and task-specific data to improve performance. 

For details on MOMENT models, training data, and experimental results, please refer to the paper [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/pdf/2402.03885.pdf).

MOMENT-1 comes in 3 sizes: [Small](https://huggingface.co/AutonLab/MOMENT-1-small), [Base](https://huggingface.co/AutonLab/MOMENT-1-base), and [Large](https://huggingface.co/AutonLab/MOMENT-1-large). 

# Usage

**Recommended Python Version:** Python 3.11 (support for additional versions is expected soon).

You can install the `momentfm` package using pip:
```bash
pip install momentfm
```
Alternatively, to install the latest version directly from the GitHub repository:
```bash
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```


To load the pre-trained model for one of the tasks, use one of the following code snippets:

**Forecasting**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': 96
    },
)
model.init()
```

**Classification**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small", 
    model_kwargs={
        'task_name': 'classification',
        'n_channels': 1,
        'num_class': 2
    },
)
model.init()
```

**Anomaly Detection, Imputation, and Pre-training**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small", 
    model_kwargs={"task_name": "reconstruction"},
)
mode.init()
```

**Representation Learning**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small", 
    model_kwargs={'task_name': 'embedding'},
)
```

### Tutorials
Here is the list of tutorials and reproducibile experiments to get started with MOMENT for various tasks:
- [Forecasting](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb)
- [Classification](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/classification.ipynb)
- [Anomaly Detection](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/anomaly_detection.ipynb)
- [Imputation](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/imputation.ipynb)
- [Representation Learning](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb)
- [Real-world Electrocardiogram (ECG) Case Study](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/ptbxl_classification.ipynb) -- This tutorial also shows how to fine-tune MOMENT for a real-world ECG classification problem, performing training and inference on multiple GPUs and parameter efficient fine-tuning (PEFT). 

## Model Details

### Model Description

- **Developed by:** [Auton Lab](https://autonlab.org/), [Carnegie Mellon University](https://www.cmu.edu/) and [University of Pennsylvania](https://www.upenn.edu/)
- **Model type:** Time-series Foundation Model
- **License:** MIT License

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/moment-timeseries-foundation-model/ (Pre-training and research code coming out soon!)
- **Paper:** https://arxiv.org/abs/2402.03885
- **Demo:** https://github.com/moment-timeseries-foundation-model/moment/tree/main/tutorials


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

We train multiple models over many days resulting in significant energy usage and a sizeable carbon footprint. However, we hope that releasing our models will ensure that future time-series modeling efforts are quicker and more efficient, resulting in lower carbon emissions.

We use the Total Graphics Power (TGP) to calculate the total power consumed for training MOMENT models, although the total power consumed by the GPU will likely vary a little based on the GPU utilization while training our model. Our calculations do not account for power demands from other sources of our compute. We use 336.566 Kg C02/MWH as the standard value of CO2 emission per megawatt hour of energy consumed for [Pittsburgh](https://emissionsindex.org/).

- **Hardware Type:** NVIDIA RTX A6000 GPU
- **GPU Hours:** 48
- **Compute Region:** Pittsburgh, USA
- **Carbon Emission (tCO2eq):** 

#### Hardware

All models were trained and evaluated on a computing cluster consisting of 128 AMD EPYC 7502 CPUs, 503 GB of RAM, and 8 NVIDIA RTX A6000 GPUs each with 49 GiB RAM. All MOMENT variants were trained on a single A6000 GPU (with any data or model parallelism).

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
If you use MOMENT please cite our paper: 

```bibtex
@inproceedings{goswami2024moment,
  title={MOMENT: A Family of Open Time-series Foundation Models},
  author={Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

**APA:**

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). 
MOMENT: A Family of Open Time-series Foundation Models. In International Conference on Machine Learning. PMLR.