from avaxtar.AvaxModel import AvaxModel

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
bearer_token = ''


if __name__ == "__main__":

	# Get the userid
	userid = ''

	# Predict
	model = AvaxModel(consumer_key, consumer_secret, access_token, access_secret, bearer_token)
	pred_proba = model.predict_from_userid_api_v1(userid)

	# Results
	print(f'User: {userid}')
	print(f'Class Probabilities: {pred_proba}')