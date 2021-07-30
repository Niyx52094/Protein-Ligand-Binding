codes built on the cpu environment.if you wish you can change it to GPU

ipynb file contains the graphes

# 1. generate the train and test data(you can skip this if you want or reproduce from the very beginning)
#put the training data into'./data/training_data'
#put the test data into'./data/testing_data'
#run:
```python main.py -lr 0.001 -dropout 0.8 -purpose preprocess -Mybest 0```

# 2. train the model
```python main.py -lr 0.001 -dropout 0.8 -purpose train -Mybest 0 ```

### you will see the accuracy and loss of valid and train stage in console window

# 3. test the performance in validation data
​	(will take approximately 4 hours,output the metrics: success rate and nDCG)

### use the model I trained
```python main.py -lr 0.001 -dropout 0.8 -purpose evaluate -Mybest 1```

### use the model you reproduced
```python main.py -lr 0.001 -dropout 0.8 -purpose evaluate -Mybest 0```

# 4. output the test file(will take approximately 6 hours.)

### use the model I trained
```python main.py -lr 0.001 -dropout 0.8 -purpose output -Mybest 1```

### use the model you reproduced
```python main.py -lr 0.001 -dropout 0.8 -purpose output -Mybest 0```