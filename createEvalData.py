#given the results of a genre tagging model, inserts markers into the text
# on which the model was tested to mark the spans the model found, currently only works for isnads
import sys
import os
import re
import json
import argparse
from tqdm import tqdm

def normalizeArabicLight(text):
	new_text = text
	new_text = re.sub("[إأٱآا]", "ا", new_text)
	new_text = re.sub("[يى]ء", "ئ", new_text)
	new_text = re.sub("ى", "ي", new_text)
	new_text = re.sub("(ؤ)", "ء", new_text)
	new_text = re.sub("(ئ)", "ء", new_text)
	return new_text

def tokenize(text):
	#remove order markers
	text = text.replace("\u202a","")
	text = text.replace("\u202c","")
	text = text.replace("\u202b","")

	arabicRegex = "[ذ١٢٣٤٥٦٧٨٩٠ّـضصثقفغعهخحجدًٌَُلإإشسيبلاتنمكطٍِلأأـئءؤرلاىةوزظْلآآ]+"

	#combine the two regexes with a | to get a regex that find both tags and arabic words
	fullRegex = arabicRegex

	tokens = [m for m in re.finditer(fullRegex,text)]

	tokenStarts = [m.start() for m in tokens]
	tokenEnds = [m.end() for m in tokens]
	tokens = [m.group() for m in tokens]

	return tokens,tokenStarts,tokenEnds

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

#extracts the sections of openITI documents, returns them as a list of strings
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

#for a given tag sequence, returns the start and end indices of any tagged portions
# note that the end index is actually
def findTaggedSpans(tags):
	i = 0
	taggedSpans = []
	inTaggedSpan = False
	begin = -1
	end = -1
	for i in range(len(tags)):
		tag = tags[i]
		if not inTaggedSpan and tag != "O":
			inTaggedSpan = True
			begin = i
		elif inTaggedSpan and tag == "O":
			inTaggedSpan = False
			end = i
			taggedSpans.append((begin,end))
		elif inTaggedSpan and "B" in tag:
			inTaggedSpan = True
			end = i
			taggedSpans.append((begin,end))
			begin = i
		elif inTaggedSpan and i == len(tags)-1:
			end = len(tags)-1
			taggedSpans.append((begin,end))
	return taggedSpans

def insertTags(text,tagData,index,taggedTokens):
	tags = tagData["tags"]
	isCorrect = tagData["knownCorrect"]

	if headless_sequences:
		requiredTags = ["I_Isnad"]
	else:
		requiredTags = ["B_Isnad","I_Isnad"]

	#if any of the required tags aren't in the tagged output, we have
	# no labeling to do so we can simply return the unaltered text
	for tag in requiredTags:
		if tag not in tags:
			return text

	#find the begin and endpoint of any tagged sections in the text
	spansToMark = findTaggedSpans(tags)
	#insert the markers where needed, beginning from the end of the text so that the positions we're adding markers to are never 
	# move by adding earlier markers
	# print(text[:100])
	# print(len(text))

	taggedTokensFromText,startsFromText,endsFromText = tokenize(text)
	assert(taggedTokensFromText==taggedTokens)
	# assert(startsFromText==tokenStarts)
	# assert(endsFromText==tokenEnds)

	tokenStarts = startsFromText
	tokenEnds = endsFromText

	
	#print(spansToMark)
	for start,end in tqdm(spansToMark[::-1],desc="Marking Spans"):
		# print("Found tagged span from %d to %d in paragraph %d (%d tokens)"%(start,end,index,len(tags)))

		startMarkerLoc = tokenStarts[start]
		endMarkerLoc = tokenEnds[end]

		#sanity check: are we tagging the right token?
		assert(text[tokenStarts[start]:tokenEnds[start]]==taggedTokens[start])

		if isCorrect:
			text = insertMarker(text,endMarkerLoc," @ISE@ ")
			text = insertMarker(text,startMarkerLoc," @ISB@ ")
		else:
			text = insertMarker(text,endMarkerLoc," @Auto_ISE@ ")
			text = insertMarker(text,startMarkerLoc," @Auto_ISB@ ")
	text = re.sub(" +"," ",text)

	return text

def insertMarker(text,index,marker):
	before = text[:index]
	after = text[index:]
	return before+marker+after

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("tagfile")
parser.add_argument("outfile")
parser.add_argument("--headless_sequences",help="If enabled, sequences with missing B tags will be included in the output.",action="store_true")
args = parser.parse_args()
infile = args.infile
tagfile = args.tagfile
outfile = args.outfile
headless_sequences = args.headless_sequences

if os.path.isdir(infile):
	filesToTag = [os.path.join(infile,f) for f in os.listdir(infile)]
	outFiles = [os.path.join(outfile,f) for f in os.listdir(infile)]
	if not os.path.exists(outfile): os.mkdir(outfile)
else:
	filesToTag = [infile]
	outFiles = [outfile]

for infile,outfile in zip(filesToTag,outFiles):
	#read in the book we're adding markers to
	book = open(infile,"r",encoding="utf8").read()

	bookID = f.split(".")
	if "-ara1" not in bookID[-1] and "-per1" not in bookID[-1]:
		bookID = bookID[:-1]
	bookID = ".".join(bookID)
	print(bookID)

	#read in the tagged output
	taggedParagraphs = [json.loads(l) for l in open(tagfile,"r",encoding="utf8") if bookID in l]

	#read in the tagged location extents so we can properly split the document
	taggedExtents = [json.loads(l) for l in open("isnadTaggedLocations.json")]

	#remove trailing spaces after the meta end tag
	book = re.sub("\#META\#Header\#End\# +\n","#META#Header#End#\n",book)
	data = book.split("#META#Header#End#\n")
	header = data[0]+"#META#Header#End#\n"
	headerLength = book.split("\n").index('#META#Header#End#')+1
	lines = [normalizeArabicLight(l) for l in data[1].split("\n")]

	if bookID in [t["text"] for t in taggedExtents]:
		correctlyTaggedSpans = [t["taggedSections"] for t in taggedExtents if t["fullURI"]==bookID][0]
	else:
		correctlyTaggedSpans = []

	#find how many lines long the header is so we can properly divide the text into documents
	headerLength = len(header.split("\n"))
	#print(headerLength)

	#iterate through the paragraphs, adding tags where needed
	paragraphs,_,_ = makeDocs(lines,correctlyTaggedSpans,headerLength,keepMarkdown=True)
	assert(len(taggedParagraphs)==len(paragraphs))

	out = open(outfile,"w",encoding="utf8")
	out.write(header)
	for i in range(len(paragraphs)):
		if taggedParagraphs[i]["knownCorrect"]:
			l = paragraphs[i].replace("@Isnad_Beg@","@ISB@").replace("@Verified_Isnad_Beg@","@ISB@").replace("@Isnad_End@","@ISE@").replace("@Verified_Isnad_End@","@ISE@")
		else:
			#last argument is for debugging
			l = insertTags(paragraphs[i],taggedParagraphs[i],i,taggedParagraphs[i]["tokens"])
		if i==len(paragraphs)-1:
			l = l.strip()
		out.write(l)