import sys
import os

project_name=sys.argv[1]
paths = ['sample-models/', 'data/', 'example/predictions/','example/structures/']

for p in paths:
	if project_name not in os.listdir(f'{p}'):
		os.mkdir(f'{p}{project_name}')