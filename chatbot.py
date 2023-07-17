import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from string import digits 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop

human_data = pd.read_table('human_text.txt', header=None, engine='python')
human_data.rename(columns={0:'human'}, inplace=True)
human_data = human_data[:1300]                        #Using only 1300 lines because of memory constraint

robot_data = pd.read_table('robot_text.txt', header=None, engine='python')
robot_data.rename(columns={0:'robot'}, inplace=True)
robot_data = robot_data[:1300] 

data = {'human':human_data.human, 'robot':robot_data.robot}
df = pd.DataFrame(data)

df.human = df.human.apply(lambda x: re.sub(r"\[\w+\]",'hi',x))      #substituting [start] with hi
df.robot = df.robot.apply(lambda x: re.sub(r"\[\w+\]",'hi',x))

df.human = df.human.apply(lambda x: x.lower())                      #converting to lower case
df.robot = df.robot.apply(lambda x: x.lower())


exclude = set(string.punctuation)
df.human = df.human.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))     #removing punctuation
df.robot = df.robot.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

remove_digits = str.maketrans('','', digits)
df.human = df.human.apply(lambda x: x.translate(remove_digits))       #removing digits
df.robot = df.robot.apply(lambda x: x.translate(remove_digits))

df.human = df.human.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))    #removing emojis from text
df.robot = df.robot.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

df.robot = df.robot.apply(lambda x: 'START' + ' ' + x + ' ' + 'END')

#getting maximum length of sentences in human text
length_list = []
for l in df.human:
  length = len(l.split())
  if length < 100:
    length_list.append(length)
max_input_length = np.max(length_list)

#truncating human sentences as per largest sentence and getting all the words in human vocab
human = []
for line in df.human:
  length = len(line.split())
  if length > max_input_length:
    line = ' '.join(line.split()[:max_input_length])
    human.append(line)
  else:
    human.append(line)
#print(human)
all_human_words = set()
for line in human:
  for word in line.split():
    if word not in all_human_words:
      all_human_words.add(word)
#print('total human words: ', len(all_human_words))


#getting maximum length of sentences in robot text
length_list = []
for l in df.robot:
  length = len(l.split())
  if length < 100:
    length_list.append(length)
max_output_length = np.max(length_list)
#print('max_output_length: ', max_output_length)


#truncating robot sentences as per largest sentence and getting all the words in robot vocab
robot = []
for line in df.robot:
  length = len(line.split())
  if length > max_output_length:
    line = ' '.join(line.split()[:max_output_length])
    robot.append(line)
  else:
    robot.append(line)
#print(robot)
all_robot_words = set()
for line in robot:
  for word in line.split():
    if word not in all_robot_words:
      all_robot_words.add(word)
#print('total robot words: ', len(all_robot_words))

input_words = sorted(list(all_human_words))
output_words = sorted(list(all_robot_words))

num_encoder_tokens = len(all_human_words)
num_decoder_tokens = len(all_robot_words)

input_token_index = dict([(word,i) for i,word in enumerate(input_words)])
output_token_index = dict([(word,i) for i,word in enumerate(output_words)])

# creating arrays of input and output data  
encoder_input_data = np.zeros((len(human), max_input_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(robot), max_output_length, num_decoder_tokens), dtype='float32')
#one hot encoding the target data as Dense layer only gives one output through softmax layer
decoder_target_data = np.zeros((len(robot), max_output_length, num_decoder_tokens))
# print(encoder_input_data.shape)
# print(decoder_input_data.shape)
# print(decoder_target_data.shape)

#putting values in input arrays and target array
for i,(input_text, output_text) in enumerate(zip(human, robot)):
  for t, word in enumerate(input_text.split()):
    #Assign 1. for the current line, timestep, & word in encoder_input_data
    encoder_input_data[i,t,input_token_index[word]] = 1
  for t, word in enumerate(output_text.split()):
    decoder_input_data[i,t,output_token_index[word]] = 1
    if t > 0:
      decoder_target_data[i,t-1,output_token_index[word]] = 1         # the target array will be one time step ahead meaning it will not contain start token

# setting hyperparameters
lstm_dim = 440

# building model for training stage
#encoder model
encoder_inputs = Input(shape=(None,num_encoder_tokens))
encoder = LSTM(lstm_dim, return_state=True)(encoder_inputs)
dropout_encoder = Dropout(0.3)
encoder_outputs, state_h, state_c = dropout_encoder(encoder)
encoder_states = [state_h, state_c]

# decoder model
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax') 
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
rmsprop = RMSprop(learning_rate=0.0001)

model.compile(optimizer=rmsprop,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'],
              sample_weight_mode='temporal')
model.summary()

checkpoint_path = './data/Ecko_chatbot'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',mode='min', save_best_only=True)
r = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=40, epochs=600, callbacks=[model_checkpoint], validation_split=0.2)
model.save('chatbot_training_model4.h5')


# model = tf.keras.models.load_model('chatbot_training_model4.h5')
#Inference Stage
#encoder model
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

#decoder model
decoder_state_input_h = Input(shape=(lstm_dim,))
decoder_state_input_c = Input(shape=(lstm_dim,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs2] + decoder_states2)

reverse_input_char_index = dict((i,char) for char, i in input_token_index.items())
reverse_output_char_index = dict((i,char) for char, i in output_token_index.items())

def decode_seq(input_seq):
  state_values = encoder_model.predict(input_seq)

  target_seq = np.zeros((1,1,num_decoder_tokens))

  target_seq[0,0,output_token_index['START']] = 1

  stop_condition = False
  decoded_sentence = ''

  while not stop_condition:
    output_tokens, h, c = decoder_model.predict([target_seq] + state_values)

    sampled_token_index = np.argmax(output_tokens[0,-1,:])
    sampled_char = reverse_output_char_index[sampled_token_index]

    decoded_sentence += ' ' + sampled_char

    if(sampled_char == 'END' or len(decoded_sentence) > max_output_length):
      stop_condition = True

    target_seq = np.zeros((1,1,num_decoder_tokens))
    target_seq[0,0,sampled_token_index] = 1

    state_values = [h,c] 

  return decoded_sentence

class Chatbot:

  negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

  #method to start conversation
  def start_chat(self):
    user_response = input("Hi, I'm a chatbot trained on random dialogs. Would you like to chat with me?\n")
    if user_response in self.negative_responses:
      print("Ok, have a great day!")
      return 
    self.chat(user_response)

  #method to handle the conversation
  def chat(self, reply):
    while not self.make_exit(reply):
      reply = input(self.generate_response(reply)+"\n")

  #method to convert user respones into matrix 
  def string_to_matrix(self, user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros((1, max_input_length, num_encoder_tokens),dtype='float32')
    for timestep, token in enumerate(tokens):
        if token in input_token_index:
          # user_input_matrix[timestep, input_features_dict[token]] = 1.
          user_input_matrix[0,timestep,input_token_index[token]] = 1
    return user_input_matrix

  #Method that will create a response using seq2seq model we built
  def generate_response(self, user_input):
      input_matrix = self.string_to_matrix(user_input)
      chatbot_response = decode_seq(input_matrix)
      #Remove <START> and <END> tokens from chatbot_response
      chatbot_response = chatbot_response.replace("START",'')
      chatbot_response = chatbot_response.replace("END",'')
      return chatbot_response

  #Method to check for exit commands
  def make_exit(self, reply):
      for exit_command in self.exit_commands:
        if exit_command in reply:
          print("Ok, have a great day!")
          return True
      return False

chatbot = Chatbot()
chatbot.start_chat()