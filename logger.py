class Logger():
	def __init__(self, filename):
		self.filename = filename
		self.clear_log()
	def clear_log(self):
		with open(self.filename, 'w') as f:
			f.write("")
	def log_info(self, message, end='\n'):
		message = "[INFO] " + message
		print(message, end=end)
		with open(self.filename, 'a') as f:
			f.write(message + end)
	def log_err(self, message, **kwargs):
		message = "[ERR] " + message
		print(message, end=end)
		with open(self.filename, 'a') as f:
			f.write(message + end)

mainLogger = Logger("logs/outputLogs.txt")