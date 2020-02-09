#given a saved CRF tagger, this script tags unseen text in markdown format and creates
# a json file with the new results
import sys
import os
import re
import json
import pickle
import argparse

from tqdm import tqdm

import sklearn_crfsuite

#for a given line and set of begin/end points for correctly tagged text,
# determines if a givne line is in one of the sections. Recall that 
# the correct extents are half open and not inclusive of their end point
def checkCorrectness(lineNumber,taggedExtents):
	if len(taggedExtents) == 0 or lineNumber > taggedExtents[-1][1]:
		return False

	for extent in taggedExtents:
		begin = extent[0]
		end = extent[1]

		if end < lineNumber:
			continue
		elif end == lineNumber:
			return False
		elif begin <= lineNumber:
			return True
		else:
			return False
	return False

#extracts the sections of openITI documents, returns them as a list of strings along with
# a list of the line indices on which the documents begin
# a list of booleans indicating whether each document has been manually tagged
def makeDocs(lines,taggedExtents,headerLength,keepMarkdown=False):
	headers = ["~~","# |","# ","### |||| ","### ||| ","### || ","### | ","### $ ","#",""]
	docs = []
	startingLines = []
	knownCorrect = []

	#set up bookkeeping
	lineNumber = headerLength
	lastCorrect = False
	for line in lines:
		lineNumber += 1

		#determine if this line is known to be correctly tagged or not
		correctlyTagged = checkCorrectness(lineNumber,taggedExtents)

		if lineNumber == headerLength+1 or correctlyTagged != lastCorrect:
			docs.append([])
			knownCorrect.append(correctlyTagged)
			startingLines.append(lineNumber)

		lineAdded = False
		for header in headers:
			if line.startswith(header):
				nextLine = line+"\n"
				docs[-1].append(nextLine)
				addedLine = True
				break #if we found a header, we're done checking headers
		if not addedLine:
			print("missing line %d\n%s"%(lineNumber,line))

		#update the status of the previous line (are we continuing a new document or making a new one?)
		lastCorrect = correctlyTagged

		#if we can discard a tagged extext since we've gotten past its endpoint, do so
		#otherwise later lines can take longer to check for correctness
		if len(taggedExtents) > 0 and lineNumber > taggedExtents[0][1]: taggedExtents = taggedExtents[1:]

	#concatenate all the lines in each document to create the full documents
	# appending to a string in a loop takes a long time when the string keeps growing
	docs = [''.join(doc) for doc in docs]

	return docs,knownCorrect,startingLines
		
def normalizeArabicLight(text):
	new_text = text
	new_text = re.sub("[إأٱآا]", "ا", new_text)
	new_text = re.sub("[يى]ء", "ئ", new_text)
	new_text = re.sub("ى", "ي", new_text)
	new_text = re.sub("(ؤ)", "ء", new_text)
	new_text = re.sub("(ئ)", "ء", new_text)
	return new_text

#given a known correct text, this function extracts tags from that text
def getTagsFromText(text):
	tags = []
	taggedTokens = []
	taggedTokenStarts = []
	taggedTokenEnds = []

	tokens,tokenStarts,tokenEnds = tokenize(text)

	inGenre =  False
	startingNew = False
	index = 0
	isnadsFound = 0
	endsFound = 0
	for token in tokens:
		#skip character order markers and new lines
		if token in ["\u202b","\u202c","\u202a","\n"] or len(token)==0:
			continue

		#check if we've started a new genre tagged span
		if token == "@Isnad_Beg@" or token == "@Verified_Isnad_Beg@" or token == "@ISB@":
			inGenre = True
			startingNew = True
			isnadsFound += 1
			# print(isnadsFound)
			# print("@Isnad_Beg@ found at %d"%index)
		elif token == "@Isnad_End@" or token == "@Verified_Isnad_End@" or token == "@ISE@":
			inGenre = False
			endsFound += 1
			# print(endsFound)
			# print("@Isnad_End@ found at %d"%index)

		if token not in ["@Isnad_Beg@","@Isnad_End@","@Verified_Isnad_Beg@","@Verified_Isnad_End@","@ISB@","@ISE@"] and len(token)>0:
			if not inGenre:
				tags.append("O")
			elif inGenre and startingNew:
				tags.append("B_Isnad")
				startingNew = False
			elif inGenre and tags[-1] != "O":
				tags.append("I_Isnad")

			taggedTokens.append(token)
			taggedTokenStarts.append(tokenStarts[index])
			taggedTokenEnds.append(tokenEnds[index])

			index += 1

	return tags,taggedTokens,taggedTokenStarts,taggedTokenEnds

#converts a list of documents into a dictionary containing the tokens that
# comprise it and the original text
def prepareText(text):
	d = {}
	d["text"] = text

	#tokenize the text and remove any tags left from previous annotation
	tokens,tokenStarts,tokenEnds = tokenize(text)
	tokens = [t for t in tokens if "@" not in t]
	d["tokens"] = tokens
	assert(len(tokens)==len(tokenStarts))

	return d

#takes a piece of text, possibly with pieces of markdown in it, and splits it into tokens
# note: this new version of this function only works with arabic. For obits and other genres
# I will need a new tokenizer, probably just a new script
def tokenize(text):
	#remove order markers
	text = text.replace("\u202a","")
	text = text.replace("\u202c","")
	text = text.replace("\u202b","")

	arabicRegex = "[ذ١٢٣٤٥٦٧٨٩٠ّـضصثقفغعهخحجدًٌَُلإإشسيبلاتنمكطٍِلأأـئءؤرلاىةوزظْلآآ]+"
	isnadTagRegex = "@Isnad_Beg@|@Isnad_End@|@Verified_Isnad_Beg@|@Verified_Isnad_End@|@ISB@|@ISE@"

	#combine the two regexes with a | to get a regex that find both tags and arabic words
	fullRegex = arabicRegex+"|"+isnadTagRegex

	tokens = [m for m in re.finditer(fullRegex,text)]

	tokenStarts = [m.start() for m in tokens]
	tokenEnds = [m.end() for m in tokens]
	tokens = [m.group() for m in tokens]

	return tokens,tokenStarts,tokenEnds

def getVocab(paragraphs):
	vocab = []
	for p in tqdm(paragraphs):
		for token in p["tokens"]:
			if token not in vocab:
				vocab.append(token)
	return vocab

def featurizeTokenCounts(vocab,windowSize=5,count=True):
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

def tagBook(f,model,featurizer):
	lines = open(f,"r",encoding="utf8").readlines()

	#extract the URI and the line number of the last metadata line
	# There really should be an easier way to do this. This is insane.
	try:
		bookID = f.split(".")
		if "-ara1" not in bookID[-1] and "-per1" not in bookID[-1]:
			bookID = bookID[:-1]
		bookID = ".".join(bookID)
	except IndexError:
		print("Cannot find URI in text %s"%f)
		sys.exit(0)
	#print(bookID)
	try:
		headerEndLine = [l for l in lines if "#META#Header#End#" in l][0]
		headerLength = lines.index(headerEndLine)+2
	except IndexError:
		print("Cannot find header endpoint in text %s"%f)
		sys.exit(0)
	#print(lines[headerLength])

	#determine which, if any, lines are tagged in the text we're looking at
	if bookID in [t["text"] for t in taggedExtents]:
		tagged = [t["taggedSections"] for t in taggedExtents if t["fullURI"]==bookID][0]
		print("Found %d tagged sections in %s"%(len(tagged),bookID))
	else:
		tagged = []

	#normalize the lines and remove metadata
	# if any extra tokens at the beginning are caught, the issue might be here.
	lines = [normalizeArabicLight(l.strip()) for l in lines if len(l.strip()) > 0 and "META" not in l and "OpenITI" not in l and "SUBJECT" not in l]
	docs,knownCorrect,_ = makeDocs(lines,tagged,headerLength)

	#tokenize the data and record whether or not each section is known to be correct
	data = []
	preprocessed = [prepareText(doc) for doc in docs]
	for i in range(len(knownCorrect)):
		preprocessed[i]["knownCorrect"] = knownCorrect[i]
		preprocessed[i]["bookID"] = bookID
		preprocessed[i]["id"] = bookID+"_"+str(i)
	data += preprocessed

	#extract features and create the data to tag
	numDocs = len(data)
	X = []
	for doc in data:
		featureVec = []
		text = doc["tokens"]

		if not doc["knownCorrect"]:
			for i in range(len(text)):
				featureVec.append(featurizer(text,i))

		X.append(featureVec)

	#use it to tag the new data
	tags = crf.predict(X)

	for i in range(numDocs):
		#if the text is known to be right, get the tags from it since the markers are still present
		if data[i]["knownCorrect"]:
			tagsFromDoc,tokens,taggedTokenStarts,taggedTokenEnds = getTagsFromText(data[i]["text"])
			data[i]["tags"] = tagsFromDoc
		else:
			data[i]["tags"] = tags[i]
			tokens = data[i]["tokens"]
			#make sure we didn't accidentally try to tag an already tagged text
			assert("@Isnad_Beg@" not in tokens and "@Isnad_End@" not in tokens and "Verified_Isnad_Beg@" not in tokens and "@Verified_Isnad_End@" not in tokens and "@ISB@" not in tokens and "@ISE@" not in tokens)

		del data[i]["text"]

		if not len(data[i]["tokens"])==len(data[i]["tags"]):
			print("Tokenization inconsistency")
			print(data[i]["knownCorrect"])
			print(len(data[i]["tokens"]))
			print(len(data[i]["tags"]))
			print(f)
			j = 0
			while j <= len(tokens)-1 and data[i]["tokens"][j]==tokens[j]:
				j+=1
			if j<len(tokens)-1:print(j,data[i]["tokens"][j],tokens[j])
			sys.exit(0)
	return data

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("model")
parser.add_argument("outfile")
args = parser.parse_args()
infile = args.infile
modelPath = args.model
outfile = args.outfile

#read in the tagged line extents
taggedExtents = [json.loads(l) for l in open("isnadTaggedLocations.json")]

if os.path.isdir(infile):
	filesToTag = [os.path.join(infile,f) for f in os.listdir(infile)]
else:
	filesToTag = [infile]

#load the model
crf = pickle.load(open(modelPath,"rb"))

vocab = []
#create the feature extracting function
featurizer = featurizeTokenCounts(vocab,windowSize=5)

#open the output file
out = open(outfile,"w",encoding="utf8")
for f in tqdm(filesToTag,desc="Tagging files"):
	results = tagBook(f,crf,featurizer)

	for result in results:
		out.write(json.dumps(result,ensure_ascii=False)+"\n")
out.close()