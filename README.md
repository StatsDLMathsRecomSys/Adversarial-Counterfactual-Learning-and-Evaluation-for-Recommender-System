# Adversarial Counterfactual Learning and Evaluationfor Recommender System


## install

Install the package in edit mode.
```{bash}
conda create -n acgan
source activate acgan
pip install -e .

#log will be saved here
mkdir -p log
```

## Data

Inside each folder in `data`, there is a Pyhton script containing the preprocessing logics. Please down the data from the links below and place them in the same folder where the script locates.

* MoivesLens-1M data: http://files.grouplens.org/datasets/movielens/ml-1m.zip

* LastFM: http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

* Goodread: https://drive.google.com/uc?id=1roQnVtWxVE1tbiXyabrotdZyUY7FA82W'


## Run experiments

* Real-world data

Example of running experiemnts on `Goodread` data.

```{bash}
SIM_PATH='./data/simulations'
DATA_PATH='./data/books'
python train_on_real.py --data_path $DATA_PATH --prefix books_real
```

* Simulation

Example of running experiemnts on `MovieLens-1M` data.

```{bash}
SIM_PATH='./data/simulations'
DATA_PATH='./data/ml-1m'

python robust_simulation.py \
--sim_path  $SIM_PATH \
--data_path $DATA_PATH --prefix ml_1m_sim

python train_on_simulation.py \
--sim_path  $SIM_PATH \
--data_path $DATA_PATH --cuda_idx 0 --prefix ml_1m_sim --models mlp acgan

```

## Validating model implementation

We use a different random split and negative sampling hence the metrics reported in the paper differ from what reported in [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) Our implementation is able to achieve similar performance using the data and evaluation methods provided in https://github.com/hexiangnan/neural_collaborative_filtering. 

To validate the performance, please extract `data/ncf_data.tar.gz` and then run `NCF_validation.py`.
