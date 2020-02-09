#reads in OpenITI data, breaks texts into paragraphs, then creates BIO tags
# for labeled isnads. Does not create tags for paragraphs after the last one containing
# marked up isnads, as we don't know where the isnads are in those sections.
import os
import sys
import re
import json
from tqdm import tqdm

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

def removeGenreTags(text):
	text = re.sub("@.*?@","",text)
	text = text.replace("\u202c","")
	text = text.replace("\u202b","")
	text = text.replace("\u202a","")
	text = re.sub(" +"," ",text)
	#somehow this process can add spaces between tildes randomly?
	text = re.sub("~ ~","~~",text)
	return text

#removes all unverified genre tags from a text
#currently only works for isnad tags, will need to be generalized
def removeUnverifiedTags(text,taggedExtents):

	lineNumber = 0
	cleanedText = []
	for line in tqdm(text.split("\n")):
		lineNumber += 1

		#if the line isn't verified, remove tags from it
		correctlyTagged = checkCorrectness(lineNumber,taggedExtents)
		if correctlyTagged:
			cleanedText += [re.sub(" +"," ",line.replace("@Isnad_Beg@","@ISB@").replace("@Isnad_End@","@ISE@").replace("@Verified_Isnad_Beg@","@ISB@").replace("@Verified_Isnad_End@","@ISE@") +"\n")]
		else:
			cleanedText += [removeGenreTags(line)+"\n"]

	#remove the final newline and return the 
	return ''.join(cleanedText).strip("\n")

#return a list of dictionaries containing the text of each paragraph
# of a text and its tags, as well as an ID
def readTexts(folder,outpath):
	#read in the information telling us what portions of the texts have correctly tagged genres
	taggedExtents = [json.loads(l) for l in open("isnadTaggedLocations.json")]
	taggedExtents = dict([(d["fullURI"],d["taggedSections"]) for d in taggedExtents])

	allData = []
	for f in os.listdir(folder):
		print(f)
		text = open(os.path.join(folder,f),encoding="utf8").read()
		bookID = f.split(".")
		if "-ara1" not in bookID[-1] and "-per1" not in bookID[-1]:
			bookID = bookID[:-1]
		bookID = ".".join(bookID)
		cleanedText = removeUnverifiedTags(text,taggedExtents[bookID])

		newFile = open(os.path.join(outpath,f),"w",encoding="utf8")
		newFile.write(cleanedText)
		newFile.close()
	return allData

path = os.path.join(os.getcwd(),"taggedTexts")
outpath = os.path.join(os.getcwd(), "taggedTexts_verifiedOnly")
data = readTexts(path,outpath)