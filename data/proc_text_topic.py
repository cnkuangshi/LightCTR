#
#  proc_text_topic.py
#  LightCTR
#
#  Created by SongKuangshi on 2017/10/15.
#  Copyright © 2017年 SongKuangshi. All rights reserved.

import os
import sys

stopset = {'a','the','of','to','an','but','or','its','about','would','and','in','that','is','are','be','been','will','this','was','for','on','as','from','at','by','with','have','which','has','had','were','it','not'}

def generate(infile,word_id_file,training_file,vocab_size):
	term_dict = {}

	infp = open(infile,'r')
	for line in infp:
		line = line.rstrip()
		if line.find('<') != -1 and line.find('>') != -1:
			continue
		info = line.split(' ')
		for term in info:
			term = term.lower()
			if term == '' or not term.isalpha() or term in stopset:
				continue
			if term.isspace() or term.find(".") != -1 or term.find(" ") != -1:
				continue
			if term in term_dict:
				term_dict[term] += 1
			else:
				term_dict[term] = 1
	term_list = sorted(term_dict.items(),key=lambda x : x[1],reverse=True)	
	print len(term_list)
	term_list = term_list[:5000]

	termid_dict = {}
	term_id = 0
	for term in term_list:
		termid_dict[term[0]] = term_id
		term_id += 1
	orderitems=[[v[1],v[0]] for v in termid_dict.items()]
	orderitems.sort()

	outfp = open(word_id_file,'w')
	for i in range(0, len(orderitems)):
		outfp.write('%d %s %d\n'%(orderitems[i][0], orderitems[i][1], term_dict[orderitems[i][1]]))
	outfp.close()

	infp.seek(0,0)
	outfp = open(training_file,'w')

	for line in infp:
		if line.find('<') != -1 and line.find('>') != -1:
			continue
		term_tf = {}
		info = line.rstrip().split(' ')
		flag = 1;
		for term in info:
			term = term.lower()
			if term not in termid_dict:
				continue
			if term in term_tf:
				term_tf[term] += 1
			else:
				term_tf[term] = 1
				flag = 0
		if flag == 1:
			continue
		out_line = ''
		for i in range(0, len(orderitems)):
			if out_line != '':
				out_line += ' '
			term = orderitems[i][1]
			if term not in term_tf:
				out_line += '0'
			else:
				out_line += '%d'%(term_tf.get(term))
		outfp.write(out_line+'\n')

	infp.close()
	outfp.close()

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print >> sys.stderr,'Usage : [%s] [input data file]'%(sys.argv[0])
		sys.exit(0)
	generate(sys.argv[1],"./vocab.txt","./train_topic.csv",sys.argv[2])
