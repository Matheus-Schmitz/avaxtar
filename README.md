# AVAXTAR: Anti-VAXx Tweet AnalyzeR

<p style='text-align: justify;'> 
AVAXTAR is a python package to identify anti-vaccine users on Twitter. The model takes a username or userID as an input and returns the probability of a user being susceptible to anti-vaccine narratives (likely to share the anti-vaccine content in the near future). The complimentary probabilities are returned in the following format: [not anti-vaccine, anti-vaccine].

The package supports both Twitter API v1 and v2. If using API v1, it requires the consumer key, consumer secret, access token and access secret. For API v2 it requires only a bearer token (If using only bearer token it accepts only userID as an input, not a screen name. In order to use screen name with API v2, API v1 keys must be passed to the model)

AVAXTAR was trained on 100GB of autolabeled twitter data. The methodology behind the package is described in full at https://arxiv.org/abs/2110.11333
</p>

## Citation

<p style='text-align: justify;'> 
If you use this code, please cite this paper:

@article{Schmitz2021,
arxivId = {2110.11333},
author = {Schmitz, Matheus and Muri{\'{c}}, Goran and Burghardt, Keith},
eprint = {2110.11333},
month = {oct},
title = {{A Python Package to Detect Anti-Vaccine Users on Twitter}},
url = {https://arxiv.org/abs/2110.11333v1},
year = {2021}
}

</p>

## Installation

<p style='text-align: justify;'> 
Attention: this package relies on a pre-trained embedding model from sent2vec (approx. 5GB). The model will be automatically downloaded when the package is first instanced in a python script, and will then be saved on the package directory for future usage.
</p>

1. Clone this repo:
```
git clone https://github.com/Matheus-Schmitz/avaxtar.git
```
2. Go to the repo's root:
```
cd avaxtar/
```
3. Install with pip:
```
pip install .
```


## Usage Example

For prediction, use:
```python
model.predict_from_userid_api_v1(userid)
```
and:
```python
model.predict_from_userid_api_v2(userid)
```

For example:
```python
from avaxtar import Avaxtar

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
bearer_token = ''


if __name__ == "__main__":

	# Get the userid
	userid = ''

	# Predict
	model = Avaxtar.AvaxModel(consumer_key, consumer_secret, access_token, access_secret, bearer_token)
	pred_proba = model.predict_from_userid_api_v1(userid)

	# Results
	print(f'User: {userid}')
	print(f'Class Probabilities: {pred_proba}')
```


## Package Details

<p style='text-align: justify;'> 
The AVAXTAR classifier is trained on a comprehensive labeled dataset that contains historical tweets of approximately 130K Twitter accounts. Each account from the dataset was assigned one out of two labels: positive for the accounts that actively spread anti-vaccination narrative \~70K and negative for the accounts that do not spread anti vaccination narrative \~60K. 

Collecting positive samples: Positive samples are gathered through a snowball method to identify a set of hashtags and keywords associated with the anti-vaccination movement, and then queried the Twitter API and collected the historical tweets of accounts that used any of the identified keywords. 

Collecting negative samples: To collect the negative samples, we first performed a mirror approach the positive samples and queried the Twitter API to get historical tweets of accounts that do not use any of the predefined keywords and hashtags. We then enlarge the number of negative samples, by gathering the tweets from accounts that are likely proponents of the vaccination. For more details about the data collection methods, please refer to: https://arxiv.org/pdf/2110.11333.pdf

After model training, we identify the optimal classification threshold to be used, based on maximizing F1 score on the validation set. We find that a threshold of 0.5938 results in the best F1 Score, and thus recommend the usage of that threshold instead of the default threshold of 0.5. Using the optimized threshold, the resulting modelwas then evaluated on a test set of users, achieving the reasonable scores, as shown in the table below.
</p>

| Metric    | Negative Class | Positive Class 	|
| ---:      |    :----:      |        :---:   	|
| Accuracy  | 0.8680 		 | 0.8680   		|
| ROC-AUC   | 0.9270         | 0.9270      		|
| PRC-AUC   | 0.8427         | 0.9677   		|
| Precision | 0.8675         | 0.8675      		|
| Recall    | 0.8680         | 0.8680   		|
| F1   		| 0.8677         | 0.8678     		|
