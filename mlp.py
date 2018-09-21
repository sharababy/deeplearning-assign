import numpy as np
import csv

X = []

with open('foo.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# next(readCSV)  # Skip header line
	for row in readCSV:
		a = list(map(float, row))
		X.append(a)


class Perceptron():
	"""docstring for Perceptron"""
	def __init__(self,num_inputs):
		super(Perceptron, self).__init__()
		
		self.num_inputs = num_inputs
		self.weights = np.random.rand(self.num_inputs)
		self.bias = np.random.rand(1)[0]
		
		self.batch_grad_w = np.zeros(self.num_inputs)
		self.batch_grad_b = 0.0
		self.batch_count = 0
		self.batch_size = 1

		self.learning_rate = 0.2

	def feedforward(self,inputs):

		output = np.sum(self.weights*inputs)+self.bias
		output = self.sigmoid(output)

		return output

	def gradient_wrt_w(self,p_grad,inputs):

		return p_grad*inputs;

	def gradient_wrt_b(self,p_grad):

		return p_grad;

	def backprop(self,inputs,p_grad):

		
		inner_work = self.sigmoid((np.sum(self.weights*np.asarray(inputs))+self.bias))
		
		grad_b = p_grad*inner_work*(1-inner_work)
		grad_w = grad_b*np.asarray(inputs)

		if self.batch_count%self.batch_size == 0 and self.batch_size != 1:
			
			self.weights = self.weights - self.learning_rate*self.batch_grad_w
			self.bias = self.bias -  self.learning_rate*self.batch_grad_b
			
			self.batch_grad_w = np.zeros(self.num_inputs)
			self.batch_grad_b = 0.0

		elif self.batch_size == 1:
			self.weights = self.weights - self.learning_rate*grad_w
			self.bias = self.bias -  self.learning_rate*grad_b

		else:
			self.batch_grad_w += grad_w
			self.batch_grad_b += grad_b

		self.batch_count += 1

		
		return p_grad*(self.weights)


	def sigmoid(self,inputs):

		return 1.0/(1.0 + np.exp(-inputs))


	def set_learning_rate(self,l_rate):

		self.learning_rate = l_rate

	def print_params(self):
		print("\tPrinting Unit Params:")
		print("\t",self.weights)
		print("\t",self.bias)

	def error(self,inputs,outputs):

		predicted = self.feedforward(inputs)

		return 0.5*((predicted - outputs)**2)





class Graph():

	def __init__(self,num_input,layer_sizes):

		self.graph = [[Perceptron(num_input) for _ in range(layer_sizes[0])]]

		for x in range(1,len(layer_sizes)):
			self.graph.append([Perceptron(layer_sizes[x-1]) for _ in range(layer_sizes[x])])
		

	def feedforward(self,inputs):

		input_holder = inputs
		layer_outputs = [input_holder]
		#  layer_outputs need to have input as index 0

		for layer in self.graph:
			layer_output = []
			for unit in layer:
				layer_output.append(unit.feedforward(input_holder))
			input_holder = np.asarray(layer_output)
			layer_outputs.append(layer_output)
			

		final_output= input_holder
		return final_output,layer_outputs


	def backprop(self,layer_outputs,expected):

		p_grad = [(layer_outputs[-1][0] - expected)]*len(self.graph[-1])
		for i in range(len(self.graph)):
			k = 0
			temp = np.asarray([0.0]*len(layer_outputs[len(self.graph)-i-1]))
			
			for unit in self.graph[len(self.graph)-i-1]:
				temp += unit.backprop(layer_outputs[len(self.graph)-i-1],p_grad[k])
				k+=1
			p_grad = temp

	def print_params(self):

		print("Printing Graph Params")
		for layer in self.graph:
			for unit in layer:
				unit.print_params()


	def error(self,inputs,outputs):

		final_pred,l = self.feedforward(inputs)
		return 0.5*((final_pred - outputs)**2)

if __name__ == "__main__":

	#  num_inputs , layer_sizes
	g = Graph(2,[4,4,1])

	epochs = 2000

	# x = X[0][0]
	# y = X[0][1]
	# o = X[0][2]

	for e in range(epochs):
		for [x,y,o] in X:

			f,l = g.feedforward([x,y])
			g.backprop(l,o)

	for [x,y,o] in X:
		f,l = g.feedforward([x,y])
		print(f,o)

	print("Total Error:")
	err = 0
	for [x,y,o] in X:
		err += g.error([x,y],o)
	print(err)
	
	




