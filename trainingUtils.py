from matplotlib import pyplot as plt
import json

def visualize_training(all_training_logs, show_training=True):
	metrics = ["validation_accuracy", "validation_loss"]
	if show_training:
		metrics += ["train_accuracy", "train_loss"]
		fig, axes = plt.subplots(2, 2)
		axes = axes.flatten()
	else:
		fig, axes = plt.subplots(2)
	for i in range(len(metrics)):
		metric = metrics[i]
		ax = axes[i]
		for j in range(len(all_training_logs)):
		  model_name, logs = list(all_training_logs.items())[j]
		  ax.plot(list(logs[metric].keys()), list(logs[metric].values()), label=model_name)
		  ax.grid(which='both')
		  ax.set_xlabel('epochs')
		  ax.set_title(metric)
		  ax.legend()

	fig.tight_layout()
	plt.show()

def save_logs(logs, filename):
	try:
		with open(filename, 'r') as f:
			logs_total = json.load(f)
	except Exception as e:
		pass

	#overwrite logs_total with new info from logs
	for logs_key in logs.keys():
		logs_total[logs_key] = logs[logs_key]

	with open(filename, 'w') as f:
		json.dump(logs_total, f)

if __name__ == '__main__':
	with open("logs/trainingLogs.json", 'r') as f:
		all_training_logs = json.load(f)
	visualize_training(all_training_logs)