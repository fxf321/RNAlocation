import os
import csv
import json
import numpy as np
from sklearn import svm
from collections import OrderedDict

def splitDataSet(filename,split_size,outdir):   #将数据集切分成指定的份数，并保存切分的结果到指定的地址
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	with open(filename,'r',encoding = 'utf-8') as fr:
		reader = csv.DictReader(fr)
		onefile = [each for each in reader] #读取data.csv 文件 dictreader读取可以用key来访问
		#print(onefile[1])
		num_line = len(onefile)
		arr = np.arange(num_line)
		np.random.shuffle(arr)
		list_all = arr.tolist()
		#print(list_all)
		each_size = (num_line+1)/split_size    #分成split_size段后每段大小
		#print(int(each_size))
		split_all = []; each_split = []
		count_num = 0; count_split = 0 	#count_num统计遍历的当前个数
										#count_split统计切分的次数
		for i in range(len(list_all)):
			each_split.append(onefile[int(list_all[i])])
			count_num += 1
			if count_num ==int(each_size):
				count_split +=1
				array_ = np.array(each_split)
				np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter='\t')  #输出每一份数据
				split_all.append(each_split) #将每一份数据加入到一个list中
				each_split = []
				count_num = 0
	return split_all

def generateDataset(dataset,outdir): #从切分的数据集中，对其中九份抽样汇成一个,\
									 #剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
	if not os.path.exists(outdir): #if not outdir,makrdir
		os.makedirs(outdir)
	train_all = []; test_all = [];cross_now = 0
	for i in range(len(dataset)):
		train_sets = []; test_sets = []; 
		cross_now += 1 #记录当前的交叉次数
		for j in range(len(dataset)):
			if i != j:#对其余九份欠抽样构成训练集
				for each_trainline in dataset[j]:
					train_sets.append(each_trainline)
		#将训练集和测试集文件单独保存起来
		with open(outdir +"/test_"+str(cross_now)+".datasets",'w') as fw_test:
			for each_testline in dataset[i]:
				test_sets.append(each_testline)
			for oneline_test in test_sets:
				fw_test.write(str(oneline_test)) #输出测试集
			test_all.append(test_sets)#保存训练集
		with open(outdir+"/train_"+str(cross_now)+".datasets",'w') as fw_train:
			for oneline_train in train_sets:
				fw_train.write(str(oneline_train))#输出训练集
			train_all.append(train_sets)#保存训练集
	return train_all,test_all

def mean_fun(onelist): #评估结果取平均
	count = 0
	for i in onelist:
		count += i
	return float(count/len(onelist))	
	
def performance(labelArr, predictArr):#性能评估，记录每次测试的指标
	accs=[];f1s=[]
	TPs = []; TNs = []; FPs = []; Performs = []
	for j in range(len(labelArr[0])):
		TP = 0.; FN = 0.; FP = 0.; TN = 0.
		for i in range(len(labelArr)):
			if labelArr[i][j] == 1 and predictArr[i][j] == 1:
				TP += 1.
			if labelArr[i] == 1 and predictArr[i] == 0:
				FN += 1.
			if labelArr[i] == 0 and predictArr[i] == 1:
				FP += 1.
			if labelArr[i] == 0 and predictArr[i] == 0:
				TN += 1.
		TPs.append(TP); FNs.append(FN); FPs.append(FP)
	Performs=[TPs,FNs,FPs]
	#计算各个rna预测的acc值
	for j in range(len(Performs[0])):
		accs[j] = Performs[0][j]/(Performs[0][j]+Performs[1][j]+Performs[2][j])
	#计算F1值
	for i in range(len(labelArr)):
		pre = 0.;rec = 0.
		for j in range(len(labelArr[0])):
			if predictArr[i][j] == 1:
				pre += Performs[0][j]/(Performs[0][j]+Performs[2][j])
			if labelArr[i][j] == 1:
				rec += Performs[0][j]/(Performs[0][j]+Performs[1][j])
		f1s[i] = 2*pre*rec/(pre+rec)
	ACC_mean = mean_fun(accs)
	F1_mean = mean_fun(f1s)
	return ACC_mean, F1_mean

def classifier(clf,train_x,train_y,test_x,test_y): #调用svm分类器，返回测试评估结果
	clf = clf.fit(train_x,train_y)
	predict = clf.predict(test_x)
	return predict

def rebuild_cor_mat(filename):
	#读json文件
	cor_dic = {}
	cor_mat = [[0.0 for col in range(1100)] for row in range(1100)] 
	with open(filename,'r') as f:           
		cor_json = json.loads(f.read())               
	#重建correlation matrices
	count = 0      
	for item in cor_json:
		if item['RNA2'] not in cor_dic:
			cor_dic[item['RNA2']]=count
			count += 1
		cor_mat[cor_dic[item['RNA1']]][cor_dic[item['RNA2']]]=item['Sim'][2]
	#print(cor_matr[3][2])
	return cor_dic, cor_mat
	
def generateLabels(dataset,arrays): # 用于生成labels
	arrays[0].append(dataset['Exosome'])
	arrays[1].append(dataset['Cytoplasm'])
	arrays[2].append(dataset['Mitochondrion'])
	arrays[3].append(dataset['Microvesicle'])
	arrays[4].append(dataset['Circulating'])
	arrays[5].append(dataset['Nucleus'])
	
def generateK_neighbors(rnaname,datanames): #用于找K个近邻
	sim = {}
	#处理miRNA关系矩阵，便于相似度查询
	rna_index, rna_corMat = rebuild_cor_mat('sim_Ours_Euc.json')
	x = rna_index[rnaname]
	for item in datanames:
		y = rna_index[item]
		if x < y:
			sim[item]=rna_corMat[x][y]
		else:
			sim[item]=rna_corMat[y][x]
	return (sorted(sim.items(), key=lambda x: x[1],  reverse=True))[:20]

def crossValidation(clf,train_all,test_all):#  交叉验证
	ACCs=[]  
	F1s=[]
	for i in range(len(train_all)):   #交叉验证len(train_all)次
		train_data = train_all[i]; train_x = []; train_y = [[]]*6
		test_data = test_all[i]; test_x = []; test_y = [[]]*6;predict_y = [[]]*6
		train_rna_name = []; test_rna_name = []
		#处理训练集和测试集数据，生成概率向量
		for eachline in train_data:    #生成训练集labels
			train_rna_name.append(eachline['name'])
			#print(eachline['Exosome'],eachline['Cytoplasm'],eachline['Mitochondrion'],eachline['Microvesicle'],eachline['Circulating'],eachline['Nucleus'])
			generateLabels(eachline,train_y)
		#print(train_rna_name)
		
		'''  #测试总表中不包含的rna
		rna_index, rna_corMat = rebuild_cor_mat('sim_Ours_Euc.json')
		print(rna_index.keys(),len(rna_index))
		print('=============================================================')
		count = 0
		for item in train_rna_name:
			if item not in rna_index.keys():
				print(item)
				count +=1
		print(count)
		'''
		
		for item in train_rna_name: 	#生成训练集向量
			k_neighbors={}; item_vec=[]
			k_neighbors=generateK_neighbors(item,train_rna_name)
			sim_sum = 0;simA=0;simB=0;simC=0;simD=0;simE=0;simF=0
			for neighbor in k_neighbors.items():
				sim_sum += neighbor[1]
				if train_y[0][train_rna_name.index(neighbor[0])] ==1 : simA+=neighbor[1]
				if train_y[1][train_rna_name.index(neighbor[0])] ==1 : simB+=neighbor[1]
				if train_y[2][train_rna_name.index(neighbor[0])] ==1 : simC+=neighbor[1]
				if train_y[3][train_rna_name.index(neighbor[0])] ==1 : simD+=neighbor[1]
				if train_y[4][train_rna_name.index(neighbor[0])] ==1 : simE+=neighbor[1]
				if train_y[5][train_rna_name.index(neighbor[0])] ==1 : simF+=neighbor[1]
			item_vec=[simA/sim_sum,simB/sim_sum,simC/sim_sum,simD/sim_sum,simE/sim_sum,simF/sim_sum]
			train_x.append(item_vec)
		for eachline in test_data:		#生成测试集的labels
			test_rna_name.append(eachline['name'])
			generateLabels(eachline,test_y)	
		for item in test_rna_name:		#生成测试集向量
			k_neighbors={};itemVector=[]
			k_neighbors=generateK_neighbors(item,train_rna_name)
			sim_sum = 0;simA=0;simB=0;simC=0;simD=0;simE=0;simF=0
			for neighbor in k_neighbors.items():
				sim_sum += neighbor[1]
				if test_y[0][test_rna_name.index(neighbor[0])] ==1 : simA+=neighbor[1]
				if test_y[1][test_rna_name.index(neighbor[0])] ==1 : simB+=neighbor[1]
				if test_y[2][test_rna_name.index(neighbor[0])] ==1 : simC+=neighbor[1]
				if test_y[3][test_rna_name.index(neighbor[0])] ==1 : simD+=neighbor[1]
				if test_y[4][test_rna_name.index(neighbor[0])] ==1 : simE+=neighbor[1]
				if test_y[5][test_rna_name.index(neighbor[0])] ==1 : simF+=neighbor[1]
			itemVector=[simA/sim_sum,simB/sim_sum,simC/sim_sum,simD/sim_sum,simE/sim_sum,simF/sim_sum]
			test_x.append(itemVector)
		#分类器对每个label训练并预测
		for i in range(6):
			predict_y.append(classifier(clf,train_x,train_y[i],test_x,test_y[i]))  #对6个位置分别分类
		ACCs[i] ,F1s[i] =performance(test_y,predict_y)
	ACC_mean = mean_fun(ACCs)
	F1_mean = mean_fun(F1s)
	return ACC_mean,F1_mean
	
if __name__ == '__main__':
	split_all = splitDataSet('data.csv',10,'splitDataSet')   #将数据切分成10份
	train_all,test_all = generateDataset(split_all,'generateDataset')  #生成训练集和测试集数据

	print("generateDataset end and cross validation start:")
	print('waiting...')
	clf = svm.SVC()
	ACC,F1 =crossValidation(clf,train_all,test_all)
	print("cross validation end")
	print('ACC =',ACC,';\n','F1 =',F1)
	
	
	