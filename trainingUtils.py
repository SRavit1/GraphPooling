from matplotlib import pyplot as plt
import json

def visualize_training(all_training_logs):
	metrics = ["accuracy", "loss"]
	fig, axes = plt.subplots(len(metrics))
	for i in range(len(metrics)):
		metric = metrics[i]
		ax = axes[i]
		for j in range(len(all_training_logs)):
		  model_name, logs = list(all_training_logs.items())[j]
		  ax.plot(list(logs[metric].keys()), list(logs[metric].values()), label=model_name)
		  ax.set_xlabel('epochs')
		  ax.set_title(metric)
		  ax.legend()

	fig.tight_layout()
	plt.show()

def save_logs(logs, filename):
	with open(filename, 'w') as f:
		json.dump(logs, f)