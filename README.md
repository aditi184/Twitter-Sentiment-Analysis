# Twitter Sentiment Analysis using Classical Machine Learning Algorithms

A sentiment categorization system for tweets is designed using classical machine learning algorithms (no deep learning). The dataset comprises of 1.6M tweets (available [here](https://www.kaggle.com/kazanova/sentiment140)) automatically labeled, and thus, noisy. This is part of [Natural Language Processing](https://www.cse.iitd.ac.in/~mausam/courses/col772/autumn2021/) course taken by [Prof Mausam](https://www.cse.iitd.ac.in/~mausam/).

The model uses ensemble learning approach. An ensemble of 5 classifiers are designed for the prediction task at hand.

## Running Mode

Training 

```bash
bash run-train.sh <data_directory> <model_directory>
```

Testing

```bash
bash run-test.sh <model_directory> <input_file_path> <output_file_path>
```
