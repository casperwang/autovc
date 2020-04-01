import new_train

save_dir = "../weights"





#Get Args:
if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type = int, default = 1000)  #How many epochs to train
	parser.add_argument('--save_dir', type = str, default = '../weights') #Save dir for weights
	parser.add_argument('--load_dir', type = str, default = '../weights/last.ckpt') #where to load weights from
	parser.add_argument('mode', type = str, default = 'train') #train for training, eval for evaluation 

	opts = parser.parse_args()
	print(opts)
