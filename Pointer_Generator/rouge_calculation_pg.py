# -*- coding: utf-8 -*-
from __future__ import print_function
from pythonrouge.pythonrouge import Pythonrouge
from pprint import pprint
import os,glob



if __name__ == '__main__':

	dir_path = "log_models/"
	ref_sum = '/rouge_ref/'
	dec_out = '/rouge_dec_dir/' ### '/rouge_sys/'
	list_files = os.listdir(dir_path)
	val_dir = []

	best_l_f_score = 0.000
	best_l_f_score_name = ""

	#### taking best model for testing

	#test_file_name = "decode_model_21000_1629329379"
	test_file_name = "decode_model_24000_1629385497"


	summary = dir_path+test_file_name+"/"+dec_out
	reference = dir_path+test_file_name+"/"+ref_sum ## check the path properly especially with slashes("/")

	P = summary
	T = reference

	#### summaries ###
	list_files_ref = sorted(os.listdir(T))
	list_files_sys = sorted(os.listdir(P))

	ref_formate = "_reference" #"000000_reference.txt"

	for i in range(len(list_files_ref)):
		fname_t = list_files_ref[i].split("_")[0]
		fname_p = list_files_sys[i].split("_")[0]
		if(fname_t == fname_p):
			#### changing the naming convention:
			os.rename(T+list_files_ref[i], T+fname_t+ref_formate+".1"+".txt") ## for reference
			os.rename(P+list_files_sys[i], P+fname_t+ref_formate+".txt") ### for decoder outputs
		else:
			print("Something went wrong with renaming the files...!!")
			break

	#f = open("testset_rouge_scores.txt","w")
	model_name = input("please enter a model name\n")

	f = open("Results_on_testset.txt","a")
	f.write(model_name+"\n")
	############################ rouge calculation #################################
	rouge = Pythonrouge(summary_file_exist=True,
		                    peer_path=summary, model_path=reference,
		                    n_gram=3, ROUGE_SU4=True, ROUGE_L=True,
		                    recall_only=False, f_measure_only=False,
		                    stemming=True, stopwords=True,
		                    word_level=True, length_limit=True, length=50,
		                    use_cf=False, cf=95, scoring_formula='average',
		                    resampling=True, samples=500, favor=True, p=0.5)
	score = rouge.calc_score()

	iteration_num = test_file_name
	iteration_num_only = test_file_name.split("_")[2]

	#print("ROUGE-1-F =%f, ROUGE-2-F = %f, ROUGE-3-F = %f, ROUGE-L-F = %f, ROUGE-SU4-F = %f " %(score['ROUGE-1-F'],score['ROUGE-2-F'],score['ROUGE-3-F'],score['ROUGE-L-F'],score['ROUGE-SU4-F']))

	#print("\n##################################################################\n\n\n")

	scores_dict = {
	"ROUGE-1": {'f':score['ROUGE-1-F'], 'p':score['ROUGE-1-P'],'r':score['ROUGE-1-R']},
	"ROUGE-2": {'f':score['ROUGE-2-F'], 'p':score['ROUGE-2-P'],'r':score['ROUGE-2-R']},
	#"ROUGE-3": {'f':score['ROUGE-3-F'], 'p':score['ROUGE-3-P'],'r':score['ROUGE-3-R']},
	"ROUGE-L": {'f':score['ROUGE-L-F'], 'p':score['ROUGE-L-P'],'r':score['ROUGE-L-R']},
	#"ROUGE-SU4": {'f':score['ROUGE-SU4-F'], 'p':score['ROUGE-SU4-P'],'r':score['ROUGE-SU4-R']}
	}
		
	f.write(test_file_name+"\n")
	f.write(str(iteration_num_only)+":\n"+str(scores_dict)+"\n\n")

	print(test_file_name+": ")
	print(scores_dict)
	print("\n")

f.close() 
