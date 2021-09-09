# AVAXTAR: Anti-VAXx Tweet AnalyzeR

AVAXTAR is a pretrained neural network pipeline that takes the screen name or user id of a twitter account, and returns how similar that user's tweets are in relation to tweets from anti-vaccine users.
The model was trained on 100GB of autolabeled twitter data, and outputs a list of probabilities associated with [not anti-vaccine, anti-vaccine].


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
