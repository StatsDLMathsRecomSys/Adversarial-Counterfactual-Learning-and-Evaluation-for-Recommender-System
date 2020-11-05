# follow the sample data preprocessing logics to prepare the data
cd '/mnt/c0r00zy/acgan/data/ml-1m'
python ml_1m.py
cd '/mnt/c0r00zy/acgan/'

SIM_PATH='/mnt/c0r00zy/acgan/data/simulations'
DATA_PATH='/mnt/c0r00zy/acgan/data/ml-1m'

python simulation.py \
--sim_path  $SIM_PATH \
--data_path $DATA_PATH --prefix ml_1m_mf

python train_on_simulation.py \
--sim_path  $SIM_PATH \
--data_path $DATA_PATH --prefix ml_1m_mf

python train_on_real.py --data_path $DATA_PATH --prefix ml_1m_real