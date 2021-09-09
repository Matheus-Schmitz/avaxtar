# AVAXTAR: Anti-VAXx Tweet AnalyzeR

AVAXTAR is a pretrained neural network pipeline that takes the screen name or user id of a twitter account, and returns how similar that user's tweets are in relation to tweets from anti-vaccine users.  
The model was trained on 100GB of autolabeled twitter data, and outputs a list of probabilities associated with [not anti-vaccine, anti-vaccine].  

The model supports both Twitter API v1 and v2. To predict with v1, the user needs its consumer key, consumer secret, access token and access secret. The v2 requires only a bearer token, but it can only predict based on user id, not on screen name. Predicting from the v2 api using screen name is only possible if v1 keys are passed to the model.  

For prediction, use:
```python
model.predict_from_userid_api_v1(userid)
```
and:
```python
model.predict_from_userid_api_v2(userid)
```


## Installation

Attention: this package relies on a pre-trained embedding model from sent2vec, with a size of 5 GB. The model will be automatically downloaded when the package is first instanced on a python script, and will then be saved on the package directory for future usage.

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

## Sample usage
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
