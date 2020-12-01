import csv
import numpy 
import math
import json


class NB:
	def __init__(self, training_file, test_file):
		self.training_file = training_file
		self.test_file = test_file
		self.seen_words = set()
		self.num_docs = 0
		self.class_data = {}

		return

	def train(self, cores=1):
		self.count_word_freqs()

		for c in self.class_data.values():
			c.log_prior = math.log(c.num_docs_in_class / self.num_docs)

			for word in self.seen_words:

				try:
					word_freq = c.word_frequencies[word]
				except:
					word_freq = 0
				
				c.log_likelihoods[word] = math.log((word_freq + 1) / (c.num_words_in_class + len(self.seen_words)))
		
		return

	def count_word_freqs(self):
		k = 0
		with open(self.training_file, mode='r',encoding="utf-8") as file:
			reader = csv.reader(file)
			#next(reader)
			for row in reader:

				if k == 0:
					k+=1
					continue
				self.num_docs += 1
				class_id = row[1]
				
				if class_id not in self.class_data.keys():
					self.class_data[class_id] = ClassData(class_id)

				self.class_data[class_id].num_docs_in_class += 1

				for word in row[0].split():
					self.seen_words.add(word)

					self.class_data[class_id].num_words_in_class += 1
					self.class_data[class_id].increment_word_freq(word)

				k +=1
				#if k >= 500:
					#return


	def test(self):
		k=0
		num_correct = 0
		num_total = 0
		with open(self.test_file, mode='r',encoding="utf-8") as file:
			reader = csv.reader(file)
			next(reader)
			for row in reader:
				num_total += 1
				correct_answer = row[1]
				s = {}
				for class_id,c in self.class_data.items():
					s[class_id] = c.log_prior

					for word in row[0].split():
						if word in self.seen_words:
							s[class_id] += c.log_likelihoods[word]

				candidate_answer = max(s.items(), key=(lambda x: x[1]))[0]
				#print(correct_answer, candidate_answer)
				if correct_answer == candidate_answer:
					num_correct += 1
				#k += 1
				#if k >= 100:
				#	break
		return num_correct / num_total

	def save_trained_weights(self,file_name):
		'''
		JSON Structure: dictionary keyed by class_id. Values are dictionaries with 'log_prior'
						and 'log_likelihoods' keys. 
						Log_prior is float.
						log_likelihoods is a dictionary keyed by words, with float values.
		'''
		json_obj = {}
		for class_id, c in self.class_data.items():
			json_obj[class_id] = {}
			json_obj[class_id]['log_prior'] = c.log_prior
			json_obj[class_id]['log_likelihoods'] = c.log_likelihoods

		with open(file_name, 'w') as output_file:
			json.dump(json_obj, output_file)



class ClassData:
	def __init__(self, id):
		self.id = id
		self.num_docs_in_class = 0
		self.num_words_in_class = 0
		self.log_prior = 0
		self.word_frequencies = {}
		self.log_likelihoods = {}
		return

	def increment_word_freq(self,word):
		if word not in self.word_frequencies.keys():
			self.word_frequencies[word] = 1
		else:
			self.word_frequencies[word] += 1
		return


model = NB(training_file='train_sanitized_nn.csv',test_file='test_sanitized_nn.csv')
model.train()
accuracy = model.test()
print('model accuracy:', accuracy)
#model.save_trained_weights('NB_weights.json')