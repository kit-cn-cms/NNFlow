from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score, roc_curve
from binary_mlp import BinaryMLP

class Classify:
	def __init__(self, n_variables, h_layers, savedir, activation, var_names = None):
		self.n_variables = n_variables
		self.n_labels = 1
		self.h_layers = h_layers
		self.activation = activation
		self.name = savedir.rsplit("/")[-1]
		self.variables = var_names
		self.savedir = savedir
		
	def classify(self, x_data):
		'''
		For external use. Load a trained net and classify external/unused data.
		
		Arguments:
		x_data: external dataset loaded with DataFrame.
			Input should be data.x
		
		Returns:
		out_values: array of output values of the trained NN ranging between 0 and 1.
		'''

		# initialize graph
		classify_graph = tf.Graph()
		
		nn = BinaryMLP(self.n_variables, self.h_layers, self.savedir, self.activation)
		# initialize graph with given shape
		with classify_graph.as_default():
			weights, biases = nn._get_parameters()
			x = tf.placeholder(tf.float32, [None, nn.n_variables])
			x_mean = tf.Variable(-1.0, validate_shape=False,  name='x_mean')
			x_std = tf.Variable(-1.0, validate_shape=False,  name='x_std')
			x_scaled = tf.div(tf.subtract(x, x_mean), x_std, name='x_scaled')
			y = tf.nn.sigmoid(nn._model(x_scaled, weights, biases))
			saver = tf.train.Saver()

			# restrict GPU use
			os.environ['CUDA_VISIBLE_DEVICES'] = "3"
			config = tf.ConfigProto()
			config.gpu_options.per_process_gpu_memory_fraction = 0.1

			with tf.Session(config = config, graph = classify_graph) as sess:
				# restore trained net
				saver.restore(sess, nn.savedir+"/"+nn.name+".ckpt")
				# calculate output values of events
				out_values = sess.run(y, {x: x_data})

			return out_values

	def plot_distribution(self, out_test, types_test, out_val, types_val, plot_loc):
		'''
		Plot a distribution of the output values of a separate sample of data 
		which was classified by a trained et.

		Arguments:
		out_values: Array of net output values for the events
		event_types: Array of flags (1.,0.) specifying the event type (signal, bkg)
		plot_loc: Location where plots should be saved	  
		'''

		# seperate signal and background into split lists:
		y1 = np.hstack((out_test, types_test))
		signal_test = y1[y1[:,1]==1, 0]
		bkg_test = y1[y1[:,1]==0, 0]

		y2 = np.hstack((out_val, types_val))
		signal_val = y2[y2[:,1]==1,0]
		bkg_val = y2[y2[:,1]==0,0]

		# weights
		sig_weights = [len(signal_test)/len(signal_val) for _ in range(len(signal_val))]
		bkg_weights = [len(bkg_test)/len(bkg_val) for _ in range(len(bkg_val))]

		# plot the distribution 
		bin_edges = np.linspace(0,1,30)
		plt.hist(bkg_val, weights = bkg_weights, bins = bin_edges, histtype = "stepfilled",
			lw = 2, label = "Background (val) *{:.2f}".format(
			len(signal_test)/len(signal_val)), color = "#d62728", alpha = 0.4)
		plt.hist(signal_val, weights = sig_weights, bins = bin_edges, histtype = "stepfilled",
			lw = 2, label = "Signal (val) *{:.2f}".format(
			len(bkg_test)/len(bkg_val)), color = "#1f77b4", alpha = 0.4)
		plt.hist(bkg_test, bins = bin_edges, histtype = "step",
			lw = 2, label = "Background (test)", color = "#d62728")
		plt.hist(signal_test, bins = bin_edges, histtype = "step",
			lw = 2, label = "Signal (test)", color = "#1f77b4")
		plt.legend(loc = "upper left")
		plt.xlabel("output value of net")
		plt.ylabel("count")
		plt.title('CMS Private Work', loc='right')
		plt.savefig(plot_loc+"dist.pdf")
		print("\ncreated distribution plot at\n\t" + plot_loc+"dist.pdf")
		plt.clf()

	def plot_roc(self, out_values, event_types, plot_loc):
		'''
		plot a ROC curve.

		Arguments:
		out_values: array of output values of the trained net.
		event_types: array of 0 and 1 Flags specifying a bkg or a signal event.
		plot_loc: final location of the plot.
		'''
		auc = roc_auc_score(event_types, out_values)
		fpr, tpr, _ = roc_curve(event_types, out_values, pos_label = 1)
		y = np.ones(len(fpr))-fpr
		plt.plot(tpr, y, label = "ROC curve, AUC = {:.4f}".format(auc),
			lw = 2, color = "red")
		plt.xlabel("accepted signal fraction", fontsize = 16)
		plt.ylabel("rejected background fraction", fontsize = 16)
		plt.legend(loc = "upper right")
		plt.xlim([0.0, 1.1])
		plt.ylim([0.0, 1.1])
		plt.grid(True)
		plt.savefig(plot_loc+"roc.pdf")
		print("\ncreated plot of roc curve at\n\t"+plot_loc+"roc.pdf")
		plt.clf()

	def plot_weights(self, y, w, plot_loc):
		sig_ind = [i for i in range(len(y)) if y[i] == 1]
		bkg_ind = [i for i in range(len(y)) if y[i] == 0]
		bins = np.linspace(min(w), max(w), 40)
		plt.hist( w[sig_ind], bins = bins, histtype = "step", lw = 2, color = "blue", label = "signal")
		plt.hist( w[bkg_ind], bins = bins, histtype = "step", lw = 2, color = "red",  label = "bkg")
		plt.xlabel("event weight product", fontsize = 16)
		plt.ylabel("count")
		plt.legend(loc = "best")
		plt.savefig(plot_loc+"weight_dist.pdf")
		print("\ncreated plot of weight distribution at \n\t"+plot_loc+"weight_dist.pdf")
		plt.clf()

	def get_cut_value(self, out_values, event_types, cut):
		'''
		get the value where a classifying cut will be set.
		
		Arguments:
		out_values: array of output values fo the trained net.
		event_types: array of 0 and 1 Flags specifying a bkg or a signal event.
		cut: two element array of type [string, float] where the first element is either 
			"signal" or "bkg" and the second element is a float between 0 and 1.
			If the first argument is "signal" the cut will be set in a way, 
			that the signal acceptance fraction is cut[1]
			if the first argument is "bkg" the cut will be set in a way, 
			that the background rejection fraction is cut[1]
		
		Returns:
		th[i]: the cut value which has the wished property.
		'''

		fpr, tpr, th = roc_curve(event_types, out_values, pos_label = 1)
		value = cut[1]
		i = 0
		if cut[0] == "signal":
			eff = tpr
			while value > eff[i]:
				i+=1
		elif cut[0] == "bkg":
			eff = np.ones(len(fpr))-fpr
			while value < eff[i]:
				i+=1

		return th[i]

	def find_best_cut(self, out, types):
		fpr, tpr, th = roc_curve(types, out, pos_label = 1)
		plt.plot(th, tpr,lw = 2,label = "signal_right pc")
		fpr = np.ones(len(fpr)) - fpr
		plt.plot(th, fpr, lw = 2,label = "backgroud_right pc")
		lsum = tpr + fpr
		qsum = np.sqrt( tpr**2 + fpr**2)
		plt.plot(th, lsum, lw = 2,label = "linear sum")
		plt.plot(th, qsum, lw = 2,label = "quadratic sum")
		plt.legend(loc = "best")
		#plt.show()
		plt.clf()
		sum = lsum
		max = 0
		index = 0
		for i in range(len(sum)):
			if sum[i] > max:
				index = i
				max = sum[i]
		return th[index]

	def cut_sample(self, labels, out_values, event_types, cut_value, return_type, plot_loc, do_print = False):
		'''
		Apply a cut to the output values of the trained net and classify the events in four categories.
		
		Arguments:
		out_values: Array of output values for every event.
		event_types: Array of signal/bkg Flags for events.
		cut_value: value between 0 and 1 to specify the applied cut.		
		return_type: "indices", "len", "array" specifying what should be returned.
		do_print: bool option to print information of cut into the console.

		Returns:
		if return_type is "indices" it returns four arrays with the indices of the four cut groups.
		if return_type is "len" it returns four integers with the lengths of the four cut groups.
		if return_type is "array" it returns four arrays with the output values of the four cut groups.
		
		first return (a_sig, true_signals): True positive signal events
		second return (a_bkg, false_signals): False negative signal events
		third return (r_bkg, true_bkg): True negative bkg events
		fourth return (a_bkg, false_bkg): False positive bkg events

		'''
		events = np.hstack((out_values, event_types))
		signal_events = events[ [i for i in range(len(events)) if events[i,0] > cut_value] ]
		bkg_events = events[ [i for i in range(len(events)) if events[i,0] < cut_value] ]

		true_signals = signal_events[ signal_events[:,1]==1, 0]
		false_signals = signal_events[ signal_events[:,1]==0, 0]

		true_bkg = bkg_events[ bkg_events[:,1]==0, 0]
		false_bkg = bkg_events[ bkg_events[:,1]==1, 0]
		with open("{}/cut_info.txt".format(plot_loc), "w") as f:
			f.write("{:<20} {:.3f}\n".format("cut value:",cut_value))
			f.write("\nCLASSIFICATION:\n\n")
			f.write("{:<20} {}\n".format(labels["bkg"],len(bkg_events)))
			f.write("{:<20} {}\n".format(labels["sig"],len(signal_events)))
			f.write("\nDISTRIBUTION:\n\n")
			f.write("{:<20} {}\n".format(labels["a_sig"],len(true_signals)))
			f.write("{:<20} {}\n".format(labels["r_sig"],len(false_bkg)))
			f.write("{:<20} {:.2f}\n\n".format(
				"right percentage :",len(true_signals)/(len(true_signals)+len(false_bkg))*100.))
			f.write("{:<20} {}\n".format(labels["r_bkg"],len(true_bkg)))
			f.write("{:<20} {}\n".format(labels["a_bkg"],len(false_bkg)))
			f.write("{:<20} {:.2f}\n\n".format(
				"right percentage :",len(true_bkg)/(len(true_bkg)+len(false_signals))*100.))

		if do_print:
			print("\n" + "-"*50 + "\n\n")
			with open("{}/cut_info.txt".format(plot_loc), "r") as f:
				for line in f:
					print(line.strip("\n"))
			print("\n" + "-"*50 + "\n\n")

		if return_type == "indices":
			accepted_ind = [i for i in range(len(events)) if events[i,0] > cut_value]
			rejected_ind = [i for i in range(len(events)) if events[i,0] < cut_value]
			a_sig = [i for i in accepted_ind if events[i,1] == 1]
			a_bkg = [i for i in accepted_ind if events[i,1] == 0]
			r_sig = [i for i in rejected_ind if events[i,1] == 1]
			r_bkg = [i for i in rejected_ind if events[i,1] == 0]
			return a_sig, a_bkg, r_bkg, r_sig
		elif return_type == "len":
			return len(true_signals), len(false_signals), len(true_bkg), len(false_bkg)
		elif return_type == "array":
			return true_signals, false_signals, true_bkg, false_bkg

	def plot_variable(self, data, weights, var_name, x_name, indices, plot_loc,label):
		'''
		Histogram the distribution of an input variable in several modes.

		Arguments:
		data: array of values of the input variable, usually extractable with x[:,i] from x of data loaded with Dataframe. i specifies the index of the variable.
		var_name: name of the variable to be histogrammed, usually extractable from vars.txt
		x_name: label of the x_axis
		indices: dictionary with four entries "acc_sig", "rej_bkg", "acc_bkg", "rej_sig" which contain an array of indices each.
		plot_loc: string specifing the final location of the plots.
		'''

		print("plotting {}".format(var_name))
		a_sig = data[ indices["acc_sig"] ]
		r_bkg = data[ indices["rej_bkg"] ]
		a_bkg = data[ indices["acc_bkg"] ]
		r_sig = data[ indices["rej_sig"] ]

		w_a_sig = weights[ indices["acc_sig"] ]
		w_r_bkg = weights[ indices["rej_bkg"] ]
		w_a_bkg = weights[ indices["acc_bkg"] ]
		w_r_sig = weights[ indices["rej_sig"] ]
		bins = np.linspace(min(data), max(data), 30)

		# plotting accepted-plot
		pl = plot_loc+ "accepted_"+var_name+".pdf"
		signal = [a_bkg, a_sig]
		signal_w = [w_a_bkg, w_a_sig]
		bkg = [r_bkg, r_sig]
		bkg_w = [w_r_bkg, w_r_sig]
		plt.hist( signal, weights = signal_w, bins = bins, lw = 2,
			histtype = "stepfilled", stacked = True, fill = True,
			label = [label["a_bkg"], label["a_sig"]], alpha = 0.9, color = ["red", "blue"])
		plt.hist( bkg, weights = bkg_w, bins = bins, lw = 2,
			histtype = "step", stacked = True,
			label = [label["r_bkg"], label["r_sig"]], color = ["magenta", "cyan"])
		plt.title(var_name, fontsize = 16)
		plt.xlabel(x_name, fontsize = 16)
		plt.ylabel("weighted count")
		plt.legend(loc = "best")
		plt.savefig(pl)
		plt.clf()
		#plotting signal-plot
		pl = plot_loc+ "signal_"+var_name+".pdf"
		ttH = [r_sig, a_sig]
		ttH_w = [w_r_sig, w_a_sig]
		ttbb = [r_bkg, a_bkg]
		ttbb_w = [w_r_bkg, w_a_bkg]
		plt.hist( ttH, weights = ttH_w, bins = bins, lw = 2,
			histtype = "stepfilled", stacked = True, fill = True,
			label = [label["r_sig"], label["a_sig"]], alpha = 0.9, color = ["red", "blue"])
		plt.hist( ttbb, weights = ttbb_w, bins = bins, lw = 2,
			histtype = "step", stacked = True,
			label = [label["r_bkg"], label["a_bkg"]], color = ["magenta", "cyan"])
		plt.title(var_name, fontsize = 16)
		plt.xlabel(x_name, fontsize = 16)
		plt.ylabel("weighted count")
		plt.legend(loc = "best")
		plt.savefig(pl)
		plt.clf()

		#plotting stacked-plot
		pl = plot_loc+ "stacked_"+var_name+".pdf"
		stack = [r_bkg, r_sig, a_bkg, a_sig]
		weights = [w_r_bkg, w_r_sig, w_a_bkg, w_a_sig]
		plt.hist( stack, weights =weights, bins = bins, lw = 2,
			histtype = "stepfilled", stacked = True, fill = True,
			label = [label["r_bkg"],label["r_sig"],label["a_bkg"],label["a_sig"]], alpha = 1.0,
			color = ["brown", "red", "blue", "yellow"])
		plt.title(var_name, fontsize = 16)
		plt.xlabel(x_name, fontsize = 16)
		plt.ylabel("weighted count")
		plt.legend(loc = "best")
		plt.savefig(pl)
		plt.clf()

		#plotting single-plot
		pl = plot_loc+ "unweighted__"+var_name+".pdf"
		plt.hist( r_bkg, bins = bins, lw = 2, histtype = "step",
			label = label["r_bkg"], color = "brown", normed = True)
		plt.hist( r_sig, bins = bins, lw = 2, histtype = "step",
			label = label["r_sig"], color = "red", normed = True)
		plt.hist( a_bkg, bins = bins, lw = 2, histtype = "step",
			label = label["a_bkg"], color = "blue", normed = True)
		plt.hist( a_sig, bins = bins, lw = 2, histtype = "step",
			label = label["a_sig"], color = "black", normed = True)
		plt.title(var_name, fontsize = 16)
		plt.xlabel(x_name, fontsize = 16)
		plt.ylabel("weighted relative count")
		plt.legend(loc = "best")
		plt.savefig(pl)
		plt.clf()
		#plotting weighted
		pl = plot_loc + "weighted_"+var_name+".pdf"
		plt.hist( a_bkg, weights = w_a_bkg, bins = bins, lw = 2, histtype = "step",
			label = label["a_bkg"], color = "blue", normed = True)
		plt.hist( a_sig, weights = w_a_sig, bins = bins, lw = 2, histtype = "step",
			label = label["a_sig"], color = "black", normed = True)
		plt.hist( r_bkg, weights = w_r_bkg, bins = bins, lw = 2, histtype = "step",
			label = label["r_bkg"], color = "brown", normed = True)
		plt.hist( r_sig, weights = w_r_sig, bins = bins, lw = 2, histtype = "step",
			label = label["r_sig"], color = "red", normed = True)
		plt.title(var_name, fontsize = 16)
		plt.xlabel(x_name, fontsize = 16)
		plt.ylabel("weighted relative count")
		plt.legend(loc = "best")
		plt.savefig(pl)
		plt.clf()


	def bash_concat_plot(self, plot_loc):
		'''
		Combine the single plots generated by plot_variable into one pdf and delete the remains.

		Arguments:
		plot_loc: location of the plots.
		'''
		print("\nconcatenating the generated plots.")
		bash_base = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="+plot_loc

		bash = bash_base+"variable_dists_accepted.pdf "+plot_loc+"accepted*.pdf"
		os.system(bash)

		bash = bash_base+"variable_dists_signal.pdf "+plot_loc+"signal*.pdf"
		os.system(bash)

		bash = bash_base+"variable_dists_stacked.pdf "+plot_loc+"stacked*.pdf"
		os.system(bash)

		bash = bash_base+"variable_dists_unweighted.pdf "+plot_loc+"unweighted*.pdf"
		os.system(bash)

		bash = bash_base+"variable_dists_weighted.pdf "+plot_loc+"weighted*.pdf"
		os.system(bash)


		os.system("rm "+plot_loc+"unweighted*.pdf")
		os.system("rm "+plot_loc+"accepted*.pdf")
		os.system("rm "+plot_loc+"signal*.pdf")
		os.system("rm "+plot_loc+"stacked*.pdf")
		os.system("rm "+plot_loc+"weighted*.pdf")
		print("\nplots saved at " + plot_loc)

	def plot_correlations(self, net1, net2):

		# plot distribution of out_values
		plt.hist2d(net1["out"],net2["out"], bins = 40, norm=LogNorm())
		plt.colorbar()
		plt.xlabel(net1["name"], fontsize = 14)
		plt.ylabel(net2["name"], fontsize = 14)
		plt.xlim([0.,1.])
		plt.ylim([0.,1.])
		corr = np.corrcoef(net1["out"], net2["out"])[0][1]
		plt.title("correlation of outputvalues of two different nets", fontsize = 13)
		text = "test AUC x = %.3f\ntest AUC y = %.3f\n\nval AUC x = %.3f\nval AUC y = %.3f\n\ncorrelation = %.3f"%(net1["t_auc"], net2["t_auc"], net1["v_auc"], net2["v_auc"], corr)
		plt.text(0.03, 0.97, text, bbox = {"facecolor":"white"}, va = "top",
			ha = "left", fontsize = 12)
		#plt.show()
		plt.savefig(net1["loc"]+"correlation.pdf")
		plt.savefig(net2["loc"]+"correlation.pdf")
		plt.clf()

		# plot distribution of signal out_values
		plt.hist2d(net1["sig"],net2["sig"], bins = 40, norm=LogNorm())
		plt.colorbar()
		plt.xlabel(net1["name"], fontsize = 14)
		plt.ylabel(net2["name"], fontsize = 14)
		plt.xlim([0.,1.])
		plt.ylim([0.,1.])
		corr = np.corrcoef(net1["sig"], net2["sig"])[0][1]
		plt.title("correlation of signal output values of two different nets", fontsize = 13)
		text = "test AUC x = %.3f\ntest AUC y = %.3f\n\nval AUC x = %.3f\nval AUC y = %.3f\n\ncorrelation = %.3f"%(net1["t_auc"], net2["t_auc"], net1["v_auc"], net2["v_auc"], corr)
		plt.text(0.03, 0.97, text, bbox = {"facecolor":"white"}, va = "top",
			ha = "left", fontsize = 12)
		#plt.show()
		plt.savefig(net1["loc"]+"sig_correlation.pdf")
		plt.savefig(net2["loc"]+"sig_correlation.pdf")
		plt.clf()

		#plot distribution of background out_values
		plt.hist2d(net1["bkg"],net2["bkg"], bins = 40, norm=LogNorm())
		plt.colorbar()
		plt.xlabel(net1["name"], fontsize = 14)
		plt.ylabel(net2["name"], fontsize = 14)
		plt.xlim([0.,1.])
		plt.ylim([0.,1.])
		corr = np.corrcoef(net1["bkg"], net2["bkg"])[0][1]
		plt.title("correlation of bkg output values of two different nets", fontsize = 13)
		text = "test AUC x = %.3f\ntest AUC y = %.3f\n\nval AUC x = %.3f\nval AUC y = %.3f\n\ncorrelation = %.3f"%(net1["t_auc"], net2["t_auc"], net1["v_auc"], net2["v_auc"], corr)
		plt.text(0.03, 0.97, text, bbox = {"facecolor":"white"}, va = "top",
			ha = "left", fontsize = 12)
		plt.savefig(net1["loc"]+"bkg_correlation.pdf")
		plt.savefig(net2["loc"]+"bkg_correlation.pdf")
		#plt.show()
		plt.clf()
		
		print("\nplots saved at\n" + net1["loc"] + "\nand\n" + net2["loc"])

	def get_auc(self, out_values, event_types):
		return roc_auc_score(event_types, out_values)

	def preselect_events(self, x_data, out_values, event_types, select_type):
		'''
		Select only bkg or signal events.

		Arguments:
		x_data: dataset loaded with DataFrame.
		out_values: Array of output values for every event in x_data.
		event_types: array of signal/bkg Flag for Data in x_data.
		select_type: Flag specifying signal (1) or bkg (0) to be selected.

		Returns:
		selected_x: cut dataset containing only bkg oder signal events.
		selected_out: net output values matching the cut dataset selected_x.
		selected_types: array of signal/bkg flags for the cut events in selected_x.

		'''
		select_indices = [ i for i in range(len(event_types)) if event_types[i] == select_type]

		selected_out = out_values[select_indices]
		selected_types = event_types[select_indices]
		selected_x = x_data[select_indices]

		return selected_x, selected_out, selected_types


	def cut_events(self, x_data, y_data, cut_value):
		'''
		Select only events with an outputvalue higher than the given argument.

		Arguments:
		x_data: dataset loaded with DataFrame.
		y_data: array of signal/bkg Flag for Data in x_data.
		cut_value: value between 0 and 1 specifying the cut.

		Returns:
		new_x_data: cut dataset matching the cut condition
		new_y_data: cut array matching the events in new_x_data.

		'''
		out_values = self.classify(x_data)
		indices_list = [i for i in range(len(out_values)) if out_values[i] > cut_value]
		new_x_data = x_data[indices_list]
		new_y_data = y_data[indices_list]

		return new_x_data, new_y_data
