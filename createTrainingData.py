# -*- coding: UTF-8 -*-
#reads in OpenITI data, then creates BIO tags for labeled genres. 
#
# Since the texts are all only partially annotated, each annotated section is split into
# training instances separately.
#
# The training examples are created by choosing an offset O between -(N-1) and 0
# for some N, then splitting the every N words, with the first chunk ending at 
# token index N-O.
#
# Several values of O are selected and each labeled section of text is chunked using each chosen offset
import os
import sys
import re
import json
import string
from io import open

def normalizeArabicLight(text):
	new_text = text
	new_text = re.sub("[إأٱآا]", "ا", new_text)
	new_text = re.sub("[يى]ء", "ئ", new_text)
	new_text = re.sub("ى", "ي", new_text)
	new_text = re.sub("(ؤ)", "ء", new_text)
	new_text = re.sub("(ئ)", "ء", new_text)
	return new_text

def removePunctuation(s):
	for c in ".,^()[]\{\}%*:-،": #add other puntuation if needed
		s = s.replace(c,"")
	return s

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

	return tokens

#for a given line and set of begin/end points for correctly tagged text,
# determines if a givne line is in one of the sections. Recall that 
# the correct extents are half open and not inclusive of their end point
def checkCorrectness(lineNumber,taggedExtents):
	if lineNumber > taggedExtents[-1][1]:
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

def isPoetry(line):
	return "%" in line

#return a list of dictionaries containing the text of each paragraph
# of a text and its tags, as well as an ID
def readTexts(folder):
	#read in the information telling us what portions of the texts have correctly tagged genres
	taggedExtents = [json.loads(l) for l in open("isnadTaggedLocations.json")]
	taggedExtents = dict([(d["fullURI"],d["taggedSections"]) for d in taggedExtents])

	allData = []
	for f in os.listdir(folder):
		text = open(os.path.join(folder,f),encoding="utf8").read()
		#assemble the URI from the filename
		bookID = f.split(".")
		if "-ara1" not in bookID[-1] and "-per1" not in bookID[-1]:
			bookID = bookID[:-1]
		bookID = ".".join(bookID)
		splitText = text.split("#META#Header#End#")
		headerLength = len(splitText[0].split("\n"))
		lines = splitText[1].split("\n")
		taggedSections,tags = getTaggedSections(lines,bookID,taggedExtents[bookID],headerLength)

		ID = 0
		for tokens,tags in zip(taggedSections,tags):
			d = {}
			assert(len(tokens)==len(tags) or len(tags)==0)
			d["tokens"] = tokens
			d["tags"] = tags
			d["bookID"] = bookID
			d["id"] = bookID+"_"+str(ID)
			d["paragraphNumber"] = ID
			ID += 1
			allData.append(d)
	return allData

#extracts the verified-correct sections of openITI documents, returns them as a list of strings
def makeDocs(lines,taggedExtents,headerLength,keepMarkdown=False):
	headers = ["~~","# |","# ","### |||| ","### ||| ","### || ","### | ","### $ ","#",""]
	docs = []
	startingLines = []
	knownCorrect = []

	#set up bookkeeping
	lineNumber = headerLength-1
	lastCorrect = False
	for line in lines:
		lineNumber += 1

		#determine if this line is known to be correctly tagged or not
		correctlyTagged = checkCorrectness(lineNumber,taggedExtents)

		if correctlyTagged and not lastCorrect:
			docs.append("")
			knownCorrect.append(correctlyTagged)
			startingLines.append(lineNumber)

		if correctlyTagged:
			lineAdded = False
			for header in headers:
				if line.startswith(header):
					if keepMarkdown:
						nextLine = line+"\n"
					else:
						nextLine = line[len(header):]+"\n"
					docs[-1] = docs[-1]+nextLine
					addedLine = True
					break #if we found a header, we're done checking headers
			if not addedLine:
				print("missing line %d\n%s"%(lineNumber,line))

		#update the status of the previous line (are we continuing a new document or making a new one?)
		lastCorrect = correctlyTagged

	return docs,knownCorrect,startingLines

def getTaggedSections(lines,bookName,taggedExtents,headerLength):

	#split text into paragraphs
	correctSections,knownCorrect,startingLines = makeDocs(lines,taggedExtents,headerLength)
	
	correctSections = [p.strip() for p in correctSections if len(p)>0]
	#remove line markers within paragraphs and normalize
	correctSections = [normalizeArabicLight(p) for p in correctSections]
	correctSections = [re.sub("\n"," \n ",p) for p in correctSections]
	
	#remove the order override characters
	correctSections = [p.replace("\u202c","") for p in correctSections]
	correctSections = [p.replace("\u202b","") for p in correctSections]
	correctSections = [p.replace("\u202a","") for p in correctSections]

	#split the sections into those that are tagged and those that aren't
	annotatedIndices = [i for i in range(len(correctSections)) if knownCorrect[i]]
	print("Found %d verified correct sections in %s"%(len(annotatedIndices),bookName))
	toTag = [(correctSections[i],i) for i in range(len(correctSections)) if i in annotatedIndices]
	untagged = [(correctSections[i],i) for i in range(len(correctSections)) if i not in annotatedIndices]


	#tag and order the sections according to their location in the document
	tags = [(tagParagraph(p),i) for (p,i) in toTag]
	taggedTokens = [(tagResults[1],i) for (tagResults,i) in tags]
	tags = [(tagResults[0],i) for (tagResults,i) in tags]
	tags += [([],[],i) for (p,i) in untagged]
	tags = [t for (t,i) in sorted(tags,key=lambda x:x[1])]
	taggedTokens = [t for (t,i) in sorted(taggedTokens,key=lambda x:x[1])]

	#TODO: ensure only words that are actually words are tagged. no punctuation, milestones,
	# page markers or anything like that

	correctSections = taggedTokens

	#debugging code for the  number of tagged tokens not matching the number of tokens from the tokenizer
	# initially it was due to the presence of order override characters (\u202a...\u202c) in the tokenized text
	# allTokens = []
	# for tokens in correctSections: allTokens += tokens 
	# allTaggedTokens = []
	# for tokens in taggedTokens: allTaggedTokens += tokens
	# for i in range(min(len(allTokens),len(allTaggedTokens))):
	# 	if allTaggedTokens[i] != allTokens[i]:
	# 		print("Nonmatching token at %d: %s, %s"%(i,allTokens[i],allTaggedTokens[i]))
	# 		print(allTokens[i])
	# 		print(len(allTokens[i]))
	# 		print(allTaggedTokens[i])
	# 		print(len(allTaggedTokens[i]))
	print("Tokens: %d"%sum([len([t for t in p]) for p in correctSections]))
	print("Token Section Lengths: %s"%[len([t for t in p]) for p in correctSections])
	print("Tagged tokens: %d"%sum([len([t for t in tagList]) for tagList in tags]))
	print("Tokens outside isnads: %d"%sum([len([t for t in tagList if t=="O"]) for tagList in tags]))
	print("Tokens beginning isnads: %d"%sum([len([t for t in tagList if t=="B_Isnad"]) for tagList in tags]))
	print("Tokens in isnads: %d"%sum([len([t for t in tagList if t=="I_Isnad"]) for tagList in tags]))

	#print(correctSections[0][:25])
	#print(tags[0][:25])
	#raise("Stop")

	return correctSections,tags

#this might not work. Test before trying to evaluate these models
def tagParagraph(text):
	tags = []
	taggedTokens = []
	tokens = tokenize(text)

	inGenre =  False
	startingNew = False
	index = 0
	isnadsFound = 0
	endsFound = 0
	for token in tokens:
		index += 1
		#skip character order markers and new lines
		if token in ["\u202b","\u202c","\u202a","\n"] or len(token)==0:
			continue

		#check if we've started a new genre tagged span
		if token in ["@Isnad_Beg@","@Verified_Isnad_Beg@","@ISB@"]:
			inGenre = True
			startingNew = True
			isnadsFound += 1
			# print(isnadsFound)
			# print("@Isnad_Beg@ found at %d"%index)
		elif token in ["@Isnad_End@","@Verified_Isnad_End@","@ISE@"]:
			inGenre = False
			endsFound += 1
			# print(endsFound)
			# print("@Isnad_End@ found at %d"%index)

		if token not in ["@Isnad_Beg@","@Isnad_End@","@Verified_Isnad_Beg@","@ISB@","@Verified_Isnad_End@","@ISE@"] and len(token)>0:
			if not inGenre:
				tags.append("O")
			elif inGenre and startingNew:
				tags.append("B_Isnad")
				startingNew = False
			elif inGenre and tags[-1] != "O":
				tags.append("I_Isnad")

			taggedTokens.append(token)

	return tags,taggedTokens

path = os.path.join(os.getcwd(),sys.argv[2])
data = readTexts(path)

outfile = sys.argv[2]
f = open(outfile,"w",encoding="utf8")
for entry in data:
	json.dump(entry,f,ensure_ascii=False)
	f.write("\n")