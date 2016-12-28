Artifitial Neural Network Library
=================================

how to build:

mkdir build

cmake ..

make

Perfomace
---------

i5-5300U & 8GB ram

| date                   | feedforward_perf | recurrent_perf | backpropagation_perf |
------------------------ | ---------------- | -------------- | -------------------- |
| FANN                   | 12.6             |                |                      |
------------------------ | ---------------- | -------------- | -------------------- |
| 2016/02/07 initial     | 8.27 sec         | 7.15 sec       | 6.00 sec             |
| 2016/02/17 AVX         | 5.53 sec         | 4.68 sec       | 4.63 sec             |
| 2016/02/17 weights     | 5.53 sec         | 4.68 sec       | 3.02 sec             |
| 2016/02/18 neuron ref. | 5.53 sec         | 4.68 sec       | 1.02 sec             |