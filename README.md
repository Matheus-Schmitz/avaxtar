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
1. Clone this repo.
2. From the root folder run:
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
