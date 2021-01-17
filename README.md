# CS-Steg

This the source code of the paper "When Steganography Meet Compressed Sensing"

![image](https://github.com/WenxueCui/CS-Steg/raw/master/images/framework.jpg)

## Requirements and Dependencies

* Ubuntu 16.04 CUDA 10.0
* Python3 （Testing in Python3.5）
* Pytorch 1.1.0   
* Torchvision 0.2.2

## Details of Implementations

In our code, two model version are included:

* Color version of CS-Steg (Hiding Color image into Color image)
* Gray version of CS-Steg (Hiding Gray image into Color image)

## How to Run

### Training CS-Steg
* Preparing the dataset for training

* Editing the path of training data in file `train_Color.py` and `train_Gray.py`.

* For color version model training in terms of subrate=0.8:

```python train_Color.py --sub_rate=0.8 --block_size=32```

* For gray version model training in terms of subrate=0.8:

```python train_Gray.py --sub_rate=0.8 --block_size=32```

### Testing CS-Steg
* Preparing the dataset for testing

* Editing the path of trained model in file `test.py` and `test_new.py`.

* For color version model testing in terms of subrate=0.8: 

```python test_Color.py --sub_rate=0.8 --block_size=32```

* For gray version model testing in terms of subrate=0.8:

```python test_Gray.py --sub_rate=0.8 --block_size=32```

## CSNet results
### Subjective results

![image](https://github.com/WenxueCui/CS-Steg/raw/master/images/results.jpg)

### Objective results
![image](https://github.com/WenxueCui/CS-Steg/raw/master/images/table.jpg)

## Additional instructions

* For training data, you can choose any natural image dataset.
* The training data is very important, if you can not achieve ideal result, maybe you can focus on the augmentation of training data or the structure of the network.
* If you like this repo, Star or Fork to support my work. Thank you.
* If you have any problem for this code, please email: wenxuecui@stu.hit.edu.cn
