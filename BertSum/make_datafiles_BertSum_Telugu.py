#coding: utf8
import os
import nltk

from wxconv import WXC

path = "data_samples/"
setnames = ["test","train","dev"]




def telugu_utf_wx(telugu_text):
	con = WXC(order='utf2wx', lang='tel')
	out = con.convert(telugu_text)
	out = out.replace('_@highlight_','@highlight')
	#print(out)
	return out


dest = "telugu_raw_stories/"
if not(os.path.exists(dest)): os.makedirs(dest)
for name in setnames:
	print("setname:",name)
	full_path = os.path.join(path, name)
	list_samples = os.listdir(full_path)
	print("list_samples: ",len(list_samples))

	for i, sample in enumerate(list_samples):
		print(sample)
		sample_full_path = os.path.join(full_path, sample)
		infile  =  sample_full_path+"/"+str(sample)+".sent.txt" #article.13.sent.txt
		outfile = sample_full_path+"/"+str(sample)+".summ.sent.txt" #article.13.summ.sent.txt
		fi = open(infile,  "r", encoding='utf-8-sig').readlines()
		fo = open(outfile, "r", encoding='utf-8-sig').readlines()##'utf-8-sig' is used to solve \uteff problem.

		article_sents =[sent.strip() for sent in fi]
		summary_sents =[sent.strip() for sent in fo]

		###################### Creating LTRCsumdata into CNNDM data formate ############3##########
		out = open(dest+name+"."+str(sample)+".story","w")
		print(article_sents, len(article_sents))
		print("\n")
		print(summary_sents, len(summary_sents))
		print("\n***********************************\n")

		#Article:###
		article_sents = [sent+"\n\n" for sent in article_sents]
		#Summary:###@highlight
		summary_sents = ["@highlight"+"\n\n"+sent for sent in summary_sents]
		cnndm_format = "".join(article_sents)+"\n\n".join(summary_sents)


		########Saving in  UTF-8 formate #########
		#out.write(cnndm_format)
		#out.close()

		#######Saving in WX format ########
		sample_wx = telugu_utf_wx(cnndm_format)
		out.write(sample_wx)
		out.close()
		#break













