from setuptools import setup, find_packages
#import subprocess

classifiers = [
	'Operating System :: OS Independent',
	'License :: OSI Approved :: MIT License',
	'Programming Language :: Python :: 3'
	]

setup(
	name='avaxtar',
	version='0.0.1',
	description='Identify if a Twitter account displays anti-vaccine sentiment.',
	py_modules=["Avaxtar",  "Avax_NN", "DF_from_DICT"],
	#package_dir={'': 'avatar'},
	author = 'Matheus Schmitz',
	author_email = 'mschmitz@usc.edu',
	packages = find_packages(),
	#package_data = {'': ['*.pt', '*.joblib', '*.bin']},
	#data_files = {'': ['*.pt', '*.joblib', '*.bin']},
	include_package_data = True,
	classifiers = classifiers,
	install_requires=[
          'numpy',
          'pandas',
          'torch',
          'sklearn',
          'tweepy',
          'requests',
          'nltk',
          'tqdm',
          'sent2vec @ git+https://github.com/epfml/sent2vec.git@master'
      ]
	)

#if name == "__main__":cd 
#subprocess.run(["pip", "install", "git+https://github.com/epfml/sent2vec.git#egg=sent2vec"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)