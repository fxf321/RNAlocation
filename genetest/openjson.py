import json

cor_dic = {}
cor_matr = [[0.0 for col in range(1100)] for row in range(1100)] 

def rebuildMat():
	#读json文件
	with open('sim_Ours_Euc.json','r') as f:           
		cor_json = json.loads(f.read())               
	#重建correlation matrices
	count = 0      
	for item in cor_json:
		if item['RNA2'] not in cor_dic:
			cor_dic[item['RNA2']]=count
			count += 1
		cor_matr[cor_dic[item['RNA1']]][cor_dic[item['RNA2']]]=item['Sim'][2]
	#print(cor_matr[3][2])

def gen_probInfo(filename,rnaname):
	#读文件 将文件中的RNA都提取出来  统计包含各个label的rna数目
	#查询相关矩阵生成dict
	#降序排序
	#生成每个对应label 的概率值，构建向量并返回


