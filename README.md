# AVATAR: Anti-VAxx Tweet AnalyzeR


## Installation
1. Install sent2vec: https://github.com/epfml/sent2vec
2. Close this repo and run:
```
pip install .
```
3. Download "wiki_unigrams.bin" from: https://drive.google.com/u/0/uc?export=download&confirm=IyJx&id=0B6VhzidiLvjSa19uYWlLUEkzX3c
4. Place the binary file on C:/Users/<User>/Anaconda3/Lib/site-packages/avatar

## Sample usage
```python
from avatar import Avatar

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
bearer_token = ''


if __name__ == "__main__":

	# Get the userid
	userid = ''

	# Predict
	model = Avatar.AvaxModel(consumer_key, consumer_secret, access_token, access_secret, bearer_token)
	pred_proba = model.predict_from_userid_api_v1(userid)

	# Results
	print(f'User: {userid}')
	print(f'Class Probabilities: {pred_proba}')
```
