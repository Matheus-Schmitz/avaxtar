# AVAXTAR: Anti-VAXx Tweet AnalyzeR

<p style='text-align: justify;'> 
AVAXTAR is a python package to identify anti-vaccine users on twitter. The model outputs complimentary probabilities for [not anti-vaccine, anti-vaccine]. AVAXTAR was trained on 100GB of autolabeled twitter data.

The model supports both Twitter API v1 and v2. To predict with v1, the user needs its consumer key, consumer secret, access token and access secret. The v2 requires only a bearer token, but it can only predict based on user id, not on screen name. Predicting from the v2 api using screen name is only possible if v1 keys are passed to the model. 

The methodology behind the package is described in full at {placeholder}
</p>

## Citation

<p style='text-align: justify;'> 
To cite this paper, please use:
{placeholder}
</p>

## Installation

<p style='text-align: justify;'> 
Attention: this package relies on a pre-trained embedding model from sent2vec, with a size of 5 GB. The model will be automatically downloaded when the package is first instanced on a python script, and will then be saved on the package directory for future usage.
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

Collecting negative samples: To collect the negative samples, we first performed a mirror approach the positive samples and queried the Twitter API to get historical tweets of accounts that do not use any of the predefined keywords and hashtags.
We then enlarge the number of negative samples, by gathering the tweets from accounts that are likely proponents of the vaccination. We identify the pro-ponents of the vaccines in the following way: First, we identify the set of twenty most prominent doctors and health experts active on Twitter. Then collected the covid-related Lists those health experts made on Twitter. From those lists, we collected approximately one thousand Twitter handles of prominent experts and doctors who tweet about the coronavirus and the pandemic. In the next step, we go through their latest 200 tweets and collected the Twitter handles of users who retweeted their tweets. That became our pool of pro-vaccine users. Finally, we collected the historical tweets of users from the pro-vaccine pool.

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
