import pandas as pd
import numpy as np
import random as rd
import argparse
import sys
import ast

from collections import Counter

# Decision strategy based on most repeated value
def decision_strategy(l):
	if len(l) == 0:
		return 2
	else:
		# Returns the most repeated value.
		return max(set(l), key = l.count)

# Obtains the previous value predicted and the amount of items used to perform the prediction.
def previous_value(dataframe, flow_id):
	if dataframe.empty:
		return (2,0)
	else:
		filter_expr = dataframe.index == flow_id
		val = dataframe.loc[filter_expr, ['prediction','delay']]
		if len(val.values) != 0:
			return (val['prediction'].iat[0], val['delay'].iat[0])
		else:
			return (2,0)

# Calculate the metrics of the classification algorithm
def erde_evaluation(merged_data, o, p, v, filename):

		try:
			# Variables
			risk_d = merged_data['prediction']
			t_risk = merged_data['truth']
			k = merged_data['delay']

			# Count of how many true positives there are
			true_pos = len(merged_data[t_risk==0])
			print("true_pos: "+str(true_pos))

			# Count of how many positive cases the system decided there were
			pos_decisions = len(merged_data[risk_d==0])
			print("pos_decisions: "+str(pos_decisions))

			# Count of how many of them are actually true positive cases
			pos_hits = len(merged_data[(t_risk==0) & (risk_d==0)])
			print("pos_hits: "+str(pos_hits))

			# Total count of users
			total_users = len(merged_data)
			print("total_users: "+str(total_users))

			# Platency values calculated based on latency (k[i]) where ref = '+' (true_pos)
			indiv_platency = []
			erde = []

			# ERDE calculus
			for i in merged_data.index: #range(total_users):

				if(risk_d[i] == 0 and t_risk[i] == 1): # FP
					erde.append(float(true_pos)/total_users)

				elif(risk_d[i] == 1 and t_risk[i] == 0): # FN
					erde.append(1.0)

				elif(risk_d[i] == 0 and t_risk[i] == 0): # TP
					erde.append(1.0 - (1.0/(1.0+np.exp(k[i]-o))))

				elif(risk_d[i] == 1 and t_risk[i] == 1): # TN
					erde.append(0.0)

				if (t_risk[i] == 0) :
					indiv_platency.append(-1 + (2 / (1 + np.exp(-p*(k[i]-1)))))

			# Calculus of F1, Precision, Recall and global ERDE
			if (pos_decisions == 0):
				precision = 0
			else:
				precision = float(pos_hits)/pos_decisions
			if (true_pos == 0):
				recall = 0
			else:
				recall = float(pos_hits)/true_pos
			if ((precision + recall) == 0):
				F1 = 0
			else:
				F1 = 2 * (precision * recall) / (precision + recall)

			# Calculus of platency
			if indiv_platency == []:
				platency_median = 0
			else:
				platency_median = np.median(indiv_platency)

			F1l = F1 * (1 - platency_median)
			erde_global = np.mean(erde) if len(erde) != 0 else 0.0 #TODO: Improve error management.

			if (filename != None):
				f = open(filename, 'w')
				old_stdout = sys.stdout
				sys.stdout = f

			# Show information
			if v:
				print('pos_hits: %i' % pos_hits)
				print('pos_decisions: %i' % pos_decisions)
				print('true_pos: %i' % true_pos)
			print('Global ERDE (with o = %d): %.4f' % (o, erde_global))
			print('F1: %.4f' % F1)
			print('F1-latency: %.4f' % F1l)
			print('P-latency: %.4f' % platency_median)
			print('Precision: %.4f' % precision)
			print('Recall: %.4f' % recall)

			if (filename != None):
				sys.stdout = old_stdout

		except:
			print('Some file or directory doesn\'t exist or an error has occurred')
			raise


parser = argparse.ArgumentParser()
parser.add_argument('-gpath', help='(Obligatory) path to golden truth file.', required=True, nargs=1, dest="gpath")
parser.add_argument('-ppath', help='(Obligatory) path to prediction file from a system.', required=True, nargs=1, dest="ppath")
parser.add_argument('-opath', help='(Optional) path to write output for each chunk. If not defined, stdout.', required=False, nargs=1, dest="opath", default=None)
parser.add_argument('-o', help='(Obligatory) o (ERDE) parameter.', required=True, nargs=1, dest="o")
parser.add_argument('-p', help='(Obligatory) p (F-latency) parameter.', required=True, nargs=1, dest="p")
parser.add_argument('-n_decision', help='(Obligatory) maximum number of items used to make the decison.', required=True, nargs=1, dest="n_decision")
parser.add_argument('-v', help='(Optional) Verbose option. Prints individual metrics values.', action='store_true')

args = parser.parse_args()

gpath = args.gpath[0]
ppath = args.ppath[0]
opath = args.opath
o = int(args.o[0])
p = float(args.p[0])
n_decision = int(args.n_decision[0])
v = args.v


# Load files
data_golden = pd.read_csv(gpath, sep="\t", header=0)
data_result = pd.read_csv(ppath, sep="\t", header=0)

# Merge tables (data) on common field 'flow_id' to compare the true risk and the decision risk
data = data_golden.merge(data_result, on='flow_id', how='right')

# ---------------------

data_groupby = data.groupby(['flow_id','truth'])

df = pd.DataFrame()

print("Execution parameters: ")
print("gpath: "+gpath)
print("ppath: "+ppath)
print("o: "+str(o)+" - "+" - p: "+str(p))
print("Point of decision: "+str(n_decision))

for i in range(1,11):
	print('\n\nChunk %i/10:' % i)

	df_list = []

	for lot in data_groupby:

		items = lot[1]['prediction'].values
		n = int(len(items) * (i * 0.1))

		flow_id = lot[0][0]
		truth = lot[0][1]

		if n != 0:
			prediction, delay = previous_value(df, flow_id)

			if prediction == 2: # In order not to change the decision once is taken.

				# Takes decision of delaying if it is not yet the number of items where the decision should be taken.
				prediction = decision_strategy(list(items[0:n])) if i>=n_decision or i==10 else 2
				# Items length
				delay = len(items[0:n])
		else:
			prediction = 2
			delay = 0


		df_list.append([flow_id, prediction, delay, truth])

	df = pd.DataFrame(df_list,columns=['id','prediction','delay','truth'])
	df = df.set_index('id') #Set index for next iteration.

	filename = None if opath == None else opath[0]+"_metrics_chunk"+str(i)+".out"

	erde_evaluation(df, o, p, v, filename)
	
