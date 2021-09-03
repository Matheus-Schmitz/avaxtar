from setuptools import setup, find_packages

classifiers = [
	'Operating System :: OS Independent',
	'License :: OSI Approved :: MIT License',
	'Programming Language :: Python :: 3'
	]

setup(
	name='AvaxModel',
	version='0.0.1',
	description='Identify if a Twitter account displays anti-vaccine sentiment.',
	py_modules=["AvaxModel"],
	#package_dir={'': 'src'},
	author = 'Matheus Schmitz',
	author_email = 'mschmitz@usc.edu',
	packages = find_packages(),
	classifiers = classifiers
	)