export THEANO_FLAGS=device=gpu0,floatX=float32

today=`date '+%Y_%m_%d__%H_%M_%S'`
python train.py  2>&1 | tee logs/final_prelu_model_$today.log
