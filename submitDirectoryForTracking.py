
import sys
import glob
import os

# move movies from bucket to tigress
movies = glob.glob(sys.argv[1] + '*')

for movie in movies:
	try:
		print(movie)
		fileName = glob.glob(movie + '/*.avi')
		if len(sys.argv) > 2:
			os.system('sbatch submitGeneric.sbatch newFlyTracker.py "' + fileName[0] + ' ' + sys.argv[2] + '"')
		else:
			os.system('sbatch submitGeneric.sbatch newFlyTracker.py ' + fileName[0])
	except:
		print('problem in ' + movie)


# move movies from tigress to scratch
