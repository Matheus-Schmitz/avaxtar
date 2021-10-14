# AVAXTAR: Anti-VAXx Tweet AnalyzeR

AVAXTAR is a python package to identify anti-vaccine users on twitter. The model outputs complimentary probabilities for [not anti-vaccine, anti-vaccine]. AVAXTAR was trained on 100GB of autolabeled twitter data.

The model supports both Twitter API v1 and v2. To predict with v1, the user needs its consumer key, consumer secret, access token and access secret. The v2 requires only a bearer token, but it can only predict based on user id, not on screen name. Predicting from the v2 api using screen name is only possible if v1 keys are passed to the model. 

The methodology behind the package is described in full at {placeholder}


## Citation

To cite this paper, please use:
{placeholder}


## Package Details

The AVAXTAR classifier is trained on a comprehensive labeled dataset that contains historical tweets of approximately 130K Twitter accounts. Each account from the dataset was assigned one out of two labels: positive for the accounts that actively spread anti-vaccination narrative \~70K and negative for the accounts that do not spread anti vaccination narrative \~60K. 
Collecting positive samples: The samples labeledpositivecomefrom an existing dataset of anti-vaccine Twitter accounts and their respective tweets, which was published by Muric et al., (2021). The authors first used a snowball method to identify a set of hashtags and keywords associated with the anti-vaccination movement, and then queried the Twitter API and collected the historical tweets of accounts that used any of the identified keywords. This way we collected more than 135 million tweets from more than 70 thousand accounts.

Collecting negative samples: To collect the negative samples, we first performed a similar approach to Muric et al., (2021) and queried the Twitter API to get historical tweets of accounts that do not use any of the predefined keywords and hashtags. That way we collected thetweets of 30𝐾 accounts that do not spread anti-vaccination narrativesand/or are impartial about the topic. By using the negative samples that most likely represent an average Twitter user, we are likely totrain a model able to differentiate between two groups of users solely based on topics or vocabulary used. To avoid that, we enlarge the number of negative samples, by gathering the tweets from accounts that are likely proponents of the vaccination. We identify the pro-ponents of the vaccines in the following way: First, we identify the set of twenty most prominent doctors and health experts active on Twitter. Then, we manually collected the URLs of Lists of those health experts they made on Twitter. We specifically searched for lists with names such as ”coronavirus experts” or ”epidemiologists”. From those lists, we collected approximately one thousand Twitter handles of prominent experts and doctors who tweet about the coronavirus and the pandemic. In the next step, we go through their latest 200 tweets and collected the Twitter handles of users who retweeted their tweets. That became our pool of pro-vaccine users. The userswho retweeted many distinct experts were more likely to be included than users who retweeted a few. Finally, we collected the historical tweets of users from the pro-vaccine pool. This way we collected more than 50 million tweets from more than 30 thousand accountsthat are most likely pro-vaccine, therefore 60𝐾 accounts and morethan 100 million tweets are gathered from users with a negative label.

Generating Training Dataset: For each account that is labeled positive we identify its labeling date as the first date in which the account published a tweet that contained one of the predefined anti-vaccination hashtags defined in Muric et al., (2021). For the negative user group, their labeling date was the date of their most recent tweet. All tweets from the 15 months proir to that date were considered, with samples being created using increasingly 90-day time windows. For each user we construct seven samples using the following time windows measured in days prior to the labeling date: [0-90), [60-150), [120-210), [180-270), [240-330), [300-390), [360-450). Time windows where the user published less than 100 tweets are ignored, to avoid generating high noise samples that could hamper model training. For each time window, all tweets from a given user were merged into a single document. The samples were then fed to a pre-trained Sent2Vec [Pagliardini et al.2018] sentence embedding model, and a 600 dimension feature vector was obtained for each sample.

Training a Classifier: The resulting training dataset with 130𝐾 samples each user embedded as 600-dimensional feature vector wasused to train the feed-forward neural network. After fine tuning the architecture and hyper-parameters, the final neural network consistsof three layers: 1) Fully connected 600-neuron layer, 2) Fully con-nected 300-neuron layer and 3) Fully connected 150-neuron layer. In between layers a 40% dropout rate was applied. We used hyper-bolic tangent activation between the layers and a softmax activationto generate prediction confidences. The batch size was 128, binary cross-entropy is used as loss function, and the optimizer is Adaptive Moment Estimation with Weight Decay (adamw) [Loshchilov andHutter 2019].

After model training, we identify the optimal classification threshold to be used, based on maximizing F1 score on the validation set. We find that a threshold of 0.5938 results in the best F1 Score, and thus recommend the usage of that threshold instead of the default threshold of 0.5. Using the optimized threshold, the resulting modelwas then evaluated on a test set of users, achieving the reasonable scores, as shown in the table below.

| Metric    | Negative Class | Positive Class 	|
| ---:      |    :----:      |        :---:   	|
| Accuracy  | 0.8680 		 | 0.8680   		|
| ROC-AUC   | 0.9270         | 0.9270      		|
| PRC-AUC   | 0.8427         | 0.9677   		|
| Precision | 0.8675         | 0.8675      		|
| Recall    | 0.8680         | 0.8680   		|
| F1   		| 0.8677         | 0.8678     		|



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
