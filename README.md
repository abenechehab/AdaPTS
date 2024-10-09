# Zero-shot Model-based Reinforcement Learning using Large Language Models

## Overview
This repository contains the official implementation of the paper:

   >Abdelhakim Benechehab, Youssef Attia El Hili, Ambroise Odonnat, Oussama Zekri, Albert Thomas, Giuseppe Paolo, Maurizio Filippone, Ievgen Redko, Balázs Kégl.
   [Zero-shot Model-based Reinforcement Learning using Large Language Models](https://arxiv.org/pdf/2402.10198).

### Abstract:
The emerging zero-shot capabilities of Large Language Models (LLMs) have led to their applications in areas extending well beyond natural language processing tasks.
In reinforcement learning, while LLMs have been extensively used in text-based environments, their integration with continuous state spaces remains understudied.
In this paper, we investigate how pre-trained LLMs can be leveraged to predict in context the dynamics of continuous Markov decision processes.
We identify handling multivariate data and incorporating the control signal as key challenges that limit the potential of LLMs' deployment in this setup and propose Disentangled In-Context Learning (DICL) to address them.
We present proof-of-concept applications in two reinforcement learning settings: model-based policy evaluation and data-augmented off-policy reinforcement learning, supported by theoretical analysis of the proposed methods.
Our experiments further demonstrate that our approach produces well-calibrated uncertainty estimates.

![main figure](figures/main_figure_for_repo.PNG)


## Directory structure
An overview of the repository's structure and contents (inside /src/dicl/):

- `main/`: Contains classes for the ICLTrainer and DICL. Objects of type ICLTrainer have methods to update the LLM context with a time series, call the LLM, collect the predicted PDFs, compute statistics, etc. Objects of type DICL have methods to fit the disentangler, predict single-step or multi-step, compute metrics (MSE and KS), and plot the results.
- `RL/`: Contains scripts for the SAC baseline and our algorithm **DICL-SAC**.
- `data/`: A sample dataset from the D4RL dataset of the HalfCheetah environment for the demo.
- `utils/`: Helper functions and classes.


## Installation

- create a conda environment:
```
conda create -n DICL python=3.9
```
- activate the environment:
```
conda activate DICL
```
- install the package
```
pip install -e .
```
- [optional] for developers, install the optional dependencies
```
pip install -e .[dev]
```
- [optional] for **DICL-SAC**, install the optional dependencies
```
pip install -e .[rl]
```

## Getting started

### DICL:
- Try our multivariate time series forecasting method (DICL) using the [getting started notebook](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/research_projects/dicl/-/blob/main/src/dicl/getting_started.ipynb).

### DICL-SAC:
- Install the RL optional dependencies then run the command *dicl-sac*.
- Example:

```
dicl-sac --seed $RANDOM --env-id Pendulum --total-timesteps 10000 --exp_name "test" --batch_size 64 --llm_batch_size 7 --llm_learning_frequency 16 --context_length 197 --interact_every 200 --learning_starts 1000 --llm_learning_starts 2000
```

- Arguments:

![main figure](figures/dicl_sac_args.PNG)

### SAC (baseline):
- Run the command *sac*.
- Example:

```
sac --seed $RANDOM --env-id "Pendulum" --total-timesteps 10000 --exp_name "test_baseline" --interact_every 200 --batch_size 64 --learning_starts 1000
```

- Arguments:

![main figure](figures/sac_args.PNG)


## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Open-source Participation

Do not hesitate to contribute to this project by submitting pull requests or issues, we would be happy to receive feedback and integrate your suggestions.

## Credits

- We would like to thank **CleanRL** for providing the SAC implementations used in this project (also **DICL-SAC** is implemented following cleanrl principles).
- We also acknowledge the work done in [LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](https://github.com/AntonioLiu97/llmICL), from which we have integrated certain [functions](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/research_projects/dicl/-/blob/main/src/dicl/utils/updated_from_liu_et_al.py) into our implementation.

---

For any questions or suggestions, feel free to contact **Abdelhakim Benechehab** at [abdelhakim.benechehab@gmail.com](mailto:abdelhakim.benechehab@gmail.com).
