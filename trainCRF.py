#given BIO tagged texts, trains a CRF to label isnads in unseen texts using token level fearures
#
#If crossval is passed as an argument, leave one out cross validation is performed and the results for each
# fold are written to a json file.
#Without crossval, the model is saved to the output location.
import sys
import json
import pickle
import argparse
import random

from tqdm import tqdm

import numpy as np

from sklearn.model_selection import cross_validate,cross_val_predict,KFold
from sklearn.metrics import make_scorer

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def getVocab(paragraphs):
	vocab = []
	for p in tqdm(paragraphs,desc="Creating vocab"):
		for token in p["tokens"]:
			if token not in vocab:
				vocab.append(token)
	return vocab

def readVectors(file):
	#TODO: Handle unknown tokens? Does glove make an UNK embedding?
	#creates the feature dictionary for the ith token of a given text
	
	#read in word embeddings
	# the filename should eventually be an argument to the program
	vectors = {}
	with open(file,"r",encoding="utf8") as infile:
		for line in infile:
			#skip the last line, if its empty
			if len(line) > 1:
				#determine the size of the vectors
				vectorList = line.split(" ")
				vectorSize = len(vectorList)-1
				term = vectorList[0]
				dims = vectorList[1:]
				vectors[term] = np.array([float(val) for val in dims])
	#print(list(vectors.keys())[:10])
	return vectors,vectorSize

#converts a long document into a series of shorter documents, repeatedly using different offsets
# each time
# in cases where the sub-document does not begin or end a complete document, the document is padded
# with dummy data to prevent the model from learning to start embedded genres with I tags
def chunkTaggedSequences(data,sampleCount,chunkSize):
	#find the offsets we will use to move the points where the documents are split
	offsets = []
	while len(offsets) < sampleCount: offsets.append(random.randint((-1*chunkSize)+1,0))

	#print(offsets)
	#print(data[0].keys())

	chunkedData = []
	documentID = 0
	#split each tag sequence up into subsequences of chunkSize beginning at [offset+chunkSize]
	for document in data:
		document["sourceDocument"] = -1
		tokens = document["tokens"]
		tags = document["tags"]
		for offset in offsets:
			currentToken = offset
			while currentToken < len(document["tokens"]):
				newDoc = {}
				newDoc["bookID"] = document["bookID"]
				newDoc["paragraphNumber"] = document["paragraphNumber"]

				newDoc["sourceDocument"] = documentID

				#actually split the tokens and tags
				startpoint = max(currentToken,0)
				endpoint = min(currentToken+chunkSize,len(tags))
				#newDoc["startIndex"] = startpoint
				#newDoc["endIndex"] = endpoint

				if startpoint > 0:
					newDoc["tags"] = ["PAD_TAG"]*windowLen
					newDoc["tokens"] = ["PAD_TOKEN"]*windowLen
				else:
					newDoc["tags"] = []
					newDoc["tokens"] = []

				newDoc["tags"] += tags[startpoint:endpoint+1]
				newDoc["tokens"] += tokens[startpoint:endpoint+1]

				if endpoint < len(tags):
					newDoc["tags"] += ["PAD_TAG"]*windowLen
					newDoc["tokens"] += ["PAD_TOKEN"]*windowLen

				currentToken += chunkSize
				
				chunkedData.append(newDoc)
		documentID += 1
	return chunkedData

def makeTrainTestSplits(numTestDocs,allDocs,trainOnChunks = False):
	splits = []
	for testIndex in range(numTestDocs):
		#get the indices of all the documents that aren't derived from the current test document or
		# are thensolves complete test documents
		if trainOnChunks:
			trainIndices = [j for j in range(len(allDocs)) if allDocs[j]["sourceDocument"] not in [testIndex,-1]]
		else:
			trainIndices = [j for j in range(len(allDocs)) if j!=testIndex]
		splits.append((trainIndices,[testIndex]))
	return splits

def featurizeTokenCounts(windowSize=5,count=True):
	#creates the feature dictionary for the ith token of a given text
	def _featureize(text,i):
		features = {}

		#add the current token
		features["token"] = text[i]

		#add features for the counts of tokens preceeding the current one
		# and those following the current one.
		# for token in vocab:
		# 	features[token+"_b"] = 0
		# 	features[token+"_a"] = 0

		windowBegin = max(i-windowSize,0)
		windowEnd = min(i+1+windowSize,len(text))

		for j in range(windowBegin,i):
			otherToken = text[j]
			featName = otherToken+"_b"
			if featName not in features:
				features[featName] = 0

			if count:
				features[featName] += 1
			else:
				features[featName] = 1

		for j in range(i+1,windowEnd):
			otherToken = text[j]
			featName = otherToken+"_a"
			if featName not in features:
				features[featName] = 0

			if count:
				features[featName] += 1
			else:
				features[featName] = 1
		return features
	return _featureize

def featurizeWordEmbeddings(vectorSize,windowSize=5):
	def _featureize(text,i):
		features = {}

		#add the current token
		#if i <=100: print("Featurizing token %s at index %d"%(text[i],i))
		if text[i] in vectors:
			tokenVector = vectors[text[i]]
		else:
			#print("missing token %s in position %d"%(text[i],i))
			tokenVector = [0.0]*vectorSize

		for j in range(len(tokenVector)):
			features["t_%d"%j] = tokenVector[j]

		windowBegin = max(i-windowSize,0)
		windowEnd = min(i+1+windowSize,len(text))

		for j in range(windowBegin,i):
			diff = i-j
			if text[j] in vectors:
				tokenVector = vectors[text[j]]
			else:
				#if j<=100: print("missing token %s in position %d"%(text[j],j))
				tokenVector = [0.0]*vectorSize
			for k in range(len(tokenVector)):
				features["t-%d_%d"%(diff,k)] = tokenVector[k]
			
		for j in range(i+1,windowEnd):
			diff = j-i
			if text[j] in vectors:
				tokenVector = vectors[text[j]]
			else:
				#if j<=100: print("missing token %s in position %d"%(text[j],j))
				tokenVector = [0.0]*vectorSize
			for k in range(len(tokenVector)):
				features["t+%d_%d"%(diff,k)] = tokenVector[k]
			
		return features
	return _featureize


parser = argparse.ArgumentParser()
parser.add_argument("--chunkSize",default=200,help="Maximum token length of training chunks",type=int)
parser.add_argument("--sampleCount",default=0,help="Number of different offsets to use",type=int)
parser.add_argument("--crossval",action='store_true')
parser.add_argument("--crossvalPredict",action='store_true')
parser.add_argument("--features",choices=["vectors","tokens"],help="Defines what type of features to use when training this model, either token frequencies or word embeddings",required=True)
parser.add_argument("--windowSize",default=5,help="The size of the window (in tokens) to look both ahead of and behind the token being featurized to creature feaures")
parser.add_argument("infile")
parser.add_argument("outfile")
args = parser.parse_args()
infile = args.infile
outfile = args.outfile
crossVal = args.crossval
crossValPredict = args.crossvalPredict
chunkSize = args.chunkSize
sampleCount = args.sampleCount
featureType = args.features
windowLen = args.windowSize

#read in the data
data = []
f = open(infile,"r",encoding="utf8")
for line in f.readlines():
	data.append(json.loads(line))

#remove untagged data
print("Read %d paragraphs"%len(data))
unchunkedData = [d for d in data if len(d["tags"])>0]
print("Read %d tagged sections"%len(data))


if sampleCount > 0:
	print("Splitting texts into multiple chunks")
	trainingData = chunkTaggedSequences(unchunkedData,sampleCount,chunkSize)
else:
	trainingData = unchunkedData

# f = open(outfile,"w",encoding="utf8")
# for entry in trainingData:
# 	json.dump(entry,f,ensure_ascii=False)
# 	f.write("\n")

#create the feature extracting function
if featureType == "tokens":
	featurizer = featurizeTokenCounts(windowSize=windowLen,count=True)
elif featureType == "vectors":
	print("Reading vectors")
	vectors,vectorSize = readVectors("vectors_priOnly.txt")
	featurizer = featurizeWordEmbeddings(vectorSize,windowSize=windowLen)
	print("Read %d vectors of size %d"%(len(vectors),vectorSize))
else:
	print("Unknown feature type: %s"%featureType)
	sys.exit(0)

#extract features and create the X and Y datasets
if sampleCount > 0:
	numDocs = len(unchunkedData)+len(trainingData)
	numTestDocs = len(unchunkedData)
	allDocs = unchunkedData+trainingData
else:
	numDocs = len(trainingData)
	numTestDocs = len(unchunkedData)
	allDocs = trainingData

X = []
Y = []
print("Featurizing documents")
for doc in tqdm(allDocs):
	featureVec = []
	text = doc["tokens"]
	tags = doc["tags"]

	for i in range(len(text)):
		featureVec.append(featurizer(text,i))

	X.append(featureVec)
	Y.append(tags)


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100
)

if crossVal:
	#this is not a good idea for large datasets, as it performs LOO cross validation
	trainTestSplits = makeTrainTestSplits(numTestDocs,allDocs,trainOnChunks = False)
	evalMetrics = {"precision":make_scorer(sklearn_crfsuite.metrics.flat_precision_score,average="macro"),
					"recall":make_scorer(sklearn_crfsuite.metrics.flat_recall_score,average="macro"),
					"f1":make_scorer(sklearn_crfsuite.metrics.flat_f1_score,average="macro"),
					"sequence_acc":make_scorer(sklearn_crfsuite.metrics.sequence_accuracy_score)}
	#this needs to be reworked to properly use only the documents not derived from the test document to 
	# train models while cross-validating
	results = cross_validate(crf,X,Y,scoring=evalMetrics,cv=trainTestSplits,n_jobs=8,return_estimator=True)
	print(results.keys())

	#add the true and predicted values for each held out sequence
	# this probably doesn't use the right held out sequence.
	# The ordering of the models is probably interleaved due to parallelism
	true = [Y[i] for i in range(numTestDocs)]
	predicted = [results["estimator"][i].predict_single(X[i]) for i in range(numTestDocs)]
	for i in range(numTestDocs): assert len(true[i])==len(predicted[i])
	results["true"] = true
	results["predicted"] = predicted

	#write the results, one per fold, to a json file
	outfile = open(outfile,"w",encoding="utf8")
	for i in range(numTestDocs):
		singleResult = {}
		trueVals = results["true"][i]
		predictedVals = results["predicted"][i]
		# print(i,len(trueVals),len(predictedVals))
		# print("True: %s"%trueVals)
		# print("Pred: %s"%predictedVals)
		singleResult["precision"] = results["test_precision"][i]
		singleResult["recall"] = results["test_recall"][i]
		singleResult["f1"] = results["test_f1"][i]
		singleResult["sequence_acc"] = results["test_sequence_acc"][i]
		singleResult["true"] = trueVals
		singleResult["predicted"] = predictedVals
		singleResult["tokens"] = data[i]["tokens"]
		# singleResult["calculated_precision"] = sklearn_crfsuite.metrics.flat_precision_score([trueVals],[predictedVals],average="macro")
		# singleResult["calculated_recall"] = sklearn_crfsuite.metrics.flat_recall_score([trueVals],[predictedVals],average="macro")
		# singleResult["calculated_f1"] = sklearn_crfsuite.metrics.flat_f1_score([trueVals],[predictedVals],average="macro")
		# singleResult["calculated_sequence_acc"] = sklearn_crfsuite.metrics.sequence_accuracy_score([trueVals],[predictedVals])

		outfile.write(json.dumps(singleResult,ensure_ascii=False)+"\n")
	outfile.close()
elif crossValPredict:
	#rather than evaluating on the folds, this will write the results from cross validation on 
	# each instance to a file using 10fold CV
	#it does this in series, and might take ages
	#results = cross_val_predict(crf,X,Y,cv=10,verbose=0,n_jobs=-1)
	
	#iterate over the folds
	#write the results, one per instance, to a json file
	outfile = open(outfile,"w",encoding="utf8")

	kf = KFold(n_splits=10,shuffle=True)
	for trainIndices,testIndices in kf.split(X,Y):
		print(len(testIndices))
		#get train and test data
		Xtrain = [X[i] for i in trainIndices]
		Xtest = [X[i] for i in testIndices]

		Ytrain = [Y[i] for i in trainIndices]

		#fit the model to the training data
		crf.fit(Xtrain,Ytrain)

		#get the results for the test data
		results = crf.predict(Xtest)

		for i,testIndex in zip(range(len(testIndices)),testIndices):
			singleResult = {}
			trueVals = Y[testIndex]
			predictedVals = results[i]
			singleResult["true"] = trueVals
			singleResult["predicted"] = predictedVals
			singleResult["tokens"] = data[testIndex]["tokens"]
			singleResult["id"] = data[testIndex]["id"]
			# singleResult["calculated_precision"] = sklearn_crfsuite.metrics.flat_precision_score([trueVals],[predictedVals],average="macro")
			# singleResult["calculated_recall"] = sklearn_crfsuite.metrics.flat_recall_score([trueVals],[predictedVals],average="macro")
			# singleResult["calculated_f1"] = sklearn_crfsuite.metrics.flat_f1_score([trueVals],[predictedVals],average="macro")
			# singleResult["calculated_sequence_acc"] = sklearn_crfsuite.metrics.sequence_accuracy_score([trueVals],[predictedVals])

			outfile.write(json.dumps(singleResult,ensure_ascii=False)+"\n")
	outfile.close()
else:
	#only fit the model to the chunked data, if it exists
	if sampleCount > 0:
		print("Fitting model to %d training examples"%len(X[numTestDocs:]))
		crf.fit(X[numTestDocs:],Y[numTestDocs:])
	#otherwise, just fit to everything
	else:
		print("Fitting model to %d training examples"%len(X))
		crf.fit(X,Y)

	print("Saving model")
	f = open(outfile,"wb")
	pickle.dump(crf,f)
	f.close()