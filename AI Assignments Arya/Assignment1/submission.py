from operator import itemgetter				#this functionality is NOT needed. It may help slightly, but you can definitely ignore it completely.

#DO NOT CHANGE!
def read_train_file():
	'''
	HELPER function: reads the training files containing the words and corresponding tags.
	Output: A tuple containing 'words' and 'tags'
	'words': This is a nested list - a list of list of words. See it as a list of sentences, with each sentence itself being a list of its words.
	For example - [['A','boy','is','running'],['Pick','the','red','cube'],['One','ring','to','rule','them','all']]
	'tags': A nested similar to above, just the corresponding tags instead of words. 
	'''						
	f = open('train','r')
	words = []
	tags = []
	lw = []
	lt = []
	for line in f:
		s = line.rstrip('\n')
		w,t= s.split('/')[0],s.split('/')[1]
		if w=='###':
			words.append(lw)
			tags.append(lt)
			lw=[]
			lt=[]
		else:
			lw.append(w)
			lt.append(t)
	words = words[1:]
	tags = tags[1:]
	assert len(words) == len(tags)
	f.close()
	return (words,tags)








#NEEDS TO BE FILLED!
def train_func(train_list_words, train_list_tags):

	'''
	This creates dictionaries storing the transition and emission probabilities - required for running Viterbi. 
	INPUT: The nested list of words and corresponding nested list of tags from the TRAINING set. This passing of correct lists and calling the function
	has been done for you. You only need to write the code for filling in the below dictionaries. (created with bigram-HMM in mind)
	OUTPUT: The two dictionaries

	HINT: Keep in mind the boundary case of the starting POS tag. You may have to choose (and stick with) some starting POS tag to compute bigram probabilities
	for the first actual POS tag.
	'''


	dict2_tag_follow_tag= {}
	"""Nested dictionary to store the transition probabilities
    each tag X is a key of the outer dictionary with an inner dictionary as the corresponding value
    The inner dictionary's key is the tag Y following X
    and the corresponding value is the number of times Y follows X - convert this count to probabilities finally before returning 
    """
	dict2_word_tag = {}
	"""Nested dictionary to store the emission probabilities.
	Each word W is a key of the outer dictionary with an inner dictionary as the corresponding value
	The inner dictionary's key is the tag X of the word W
	and the corresponding value is the number of times X is a tag of W - convert this count to probabilities finally before returning
	"""


	#      *** WRITE YOUR CODE HERE ***    
	tags = {}
	tags['###'] = 0
	for s in train_list_tags: #calculate array of number of tags repeated..later used in ems.prob
		for tag in s:
			if tag in tags.keys():
				tags[tag] += 1
			else:
				tags[tag] = 1
		tags['###'] += 1

	for s in train_list_tags:
		prev = '###'
		for tag in s:
			if prev in dict2_tag_follow_tag.keys():
				if tag in dict2_tag_follow_tag[prev].keys():
					dict2_tag_follow_tag[prev][tag] += 1 / tags[prev]
				else:
					dict2_tag_follow_tag[prev][tag] = 1 / tags[prev]
			else:
				dict2_tag_follow_tag[prev] = {}
				dict2_tag_follow_tag[prev][tag] = 1 / tags[prev]
			prev = tag

	c = 0
	for s in train_list_words:
		tag_seq = train_list_tags[c]
		for word, tag in zip(s, tag_seq):
			if word in dict2_word_tag.keys():
				if tag in dict2_word_tag[word].keys():
					dict2_word_tag[word][tag] += 1 / tags[tag]
				else:
					dict2_word_tag[word][tag] = 1 / tags[tag]
			else:
				dict2_word_tag[word] = {}
				dict2_word_tag[word][tag] = 1 / tags[tag]
		c += 1




	return (dict2_tag_follow_tag, dict2_word_tag)



#NEEDS TO BE FILLED!
def assign_POS_tags(test_words, dict2_tag_follow_tag, dict2_word_tag):

	'''
	This is where you write the actual code for Viterbi algorithm. 
	INPUT: test_words - this is a nested list of words for the TEST set
	       dict2_tag_follow_tag - the transition probabilities (bigram), filled in by YOUR code in the train_func
	       dict2_word_tag - the emission probabilities (bigram), filled in by YOUR code in the train_func
	OUTPUT: a nested list of predicted tags corresponding to the input list test_words. This is the 'output_test_tags' list created below, and returned after your code
	ends.

	HINT: Keep in mind the boundary case of the starting POS tag. You will have to use the tag you created in the previous function here, to get the
	transition probabilities for the first tag of sentence...
	HINT: You need not apply sophisticated smoothing techniques for this particular assignment.
	If you cannot find a word in the test set with probabilities in the training set, simply tag it as 'N'. 
	So if you are unable to generate a tag for some word due to unavailibity of probabilities from the training set,
	just predict 'N' for that word.

	'''



	output_test_tags = []    #list of list of predicted tags, corresponding to the list of list of words in Test set (test_words input to this function)


	#      *** WRITE YOUR CODE HERE *** 
	output_test_tags = [[] for x in range(len(test_words))]
	# print(output_test_tags)
	#unk = 0
	#w=0
	k = 0
	for s in test_words:
		v = [[0 for x in range(len(s) + 1)] for y in range(len(dict2_tag_follow_tag))]
		prev_tag = "###"
		i = 0
		for x in range(len(dict2_tag_follow_tag)):   #making viterbi matrix
			v[x][0] = 0
		# print(v)
		back = 1
		max_cur_tag = "START"
		for word in s:
			#w += 1
			if word not in dict2_word_tag.keys():
				output_test_tags[k].append('N')#unknown words given N (noun) tag
				prev_tag = 'N'
				back = 1
				#unk += 1
				continue
			i += 1
			j = 0
			max_cur = 0

			for tag in dict2_tag_follow_tag:
				if tag not in dict2_word_tag[word].keys():
					emission_prob = 0
				else:
					emission_prob = dict2_word_tag[word][tag]

				if prev_tag == "###":
					transition_prob = 1
				else:
					if tag not in dict2_tag_follow_tag[prev_tag].keys():
						transition_prob = 0
					else:
						transition_prob = dict2_tag_follow_tag[prev_tag][tag]

				v[j][i] = back * transition_prob * emission_prob
				if v[j][i] > max_cur:
					max_cur = v[j][i]
					max_cur_tag = tag
				j += 1

			prev_tag = max_cur_tag
			output_test_tags[k].append(max_cur_tag)
			back = max_cur
		k += 1
	#print("OUTPUT TAGS:\n{}".format(output_test_tags))
	# print(v)
	#print(unk)
	#print(unk/w)
	#print(dict2_tag_follow_tag)
	#print(dict2_word_tag)
	# END OF YOUR CODE

	return output_test_tags








# DO NOT CHANGE!
def public_test(predicted_tags):
	'''
	HELPER function: Takes in the nested list of predicted tags on test set (prodcuced by the assign_POS_tags function above)
	and computes accuracy on the public test set. Note that this accuracy is just for you to gauge the correctness of your code.
	Actual performance will be judged on the full test set by the TAs, using the output file generated when your code runs successfully.
	'''

	f = open('test_public_labeled','r')
	words = []
	tags = []
	lw = []
	lt = []
	for line in f:
		s = line.rstrip('\n')
		w,t= s.split('/')[0],s.split('/')[1]
		if w=='###':
			words.append(lw)
			tags.append(lt)
			lw=[]
			lt=[]
		else:
			lw.append(w)
			lt.append(t)
	words = words[1:]
	tags = tags[1:]
	assert len(words) == len(tags)
	f.close()
	public_predictions = predicted_tags[:len(tags)]
	assert len(public_predictions)==len(tags)

	correct = 0
	total = 0
	flattened_actual_tags = []
	flattened_pred_tags = []
	for i in range(len(tags)):
		x = tags[i]
		y = public_predictions[i]
		if len(x)!=len(y):
			print(i)
			print(x)
			print(y)
			break
		flattened_actual_tags+=x
		flattened_pred_tags+=y
	assert len(flattened_actual_tags)==len(flattened_pred_tags)
	correct = 0.0
	for i in range(len(flattened_pred_tags)):
		if flattened_pred_tags[i]==flattened_actual_tags[i]:
			correct+=1.0
	print('Accuracy on the Public set = '+str(correct/len(flattened_pred_tags)))



# DO NOT CHANGE!
def evaluate():
	words_list_train = read_train_file()[0]
	tags_list_train = read_train_file()[1]

	dict2_tag_tag = train_func(words_list_train,tags_list_train)[0]
	dict2_word_tag = train_func(words_list_train,tags_list_train)[1]

	f = open('test_full_unlabeled','r')

	words = []
	l=[]
	for line in f:
		w = line.rstrip('\n')
		if w=='###':
			words.append(l)
			l=[]
		else:
			l.append(w)
	f.close()
	words = words[1:]
	test_tags = assign_POS_tags(words, dict2_tag_tag, dict2_word_tag)
	assert len(words)==len(test_tags)

	public_test(test_tags)

	#create output file with all tag predictions on the full test set

	f = open('output','w')
	f.write('###/###\n')
	for i in range(len(words)):
		sent = words[i]
		pred_tags = test_tags[i]
		for j in range(len(sent)):
			word = sent[j]
			pred_tag = pred_tags[j]
			f.write(word+'/'+pred_tag)
			f.write('\n')
		f.write('###/###\n')
	f.close()

	print('OUTPUT file has been created')

if __name__ == "__main__":
	evaluate()

