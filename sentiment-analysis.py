# Jonathan Austin & Hrishi Mukherjee
# 100942636       & 100888108

from keras.layers import Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import argparse

####################
# USED FOR RUNNING TENSORFLOW ON GPU 
# CODE USED: https://github.com/tensorflow/models/blob/master/tutorials/image
# /cifar10/cifar10_multi_gpu_train.py
####################
parser = argparse.ArgumentParser()
parser.add_argument('--log_device_placement', type=bool, default=False,

                    help='Whether to log device placement.')
FLAGS = parser.parse_args()

####################
# INITIAL PARAMETERS
####################
MAX_SEQUENCE_LENGTH = 283
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
batch_size = 16
test_size = 200

####################
# FUNCTIONS
####################
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
	
def model(X, w, w2, w3, w_fc, w_o, p_keep_conv, p_keep_hidden, act=tf.nn.relu):
	# Model Function with Convolutional Layers

    # Run GloVe embedding layer on data
    with tf.name_scope('embedding_layer'):
        embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)
		
		#embedded_sequences shape=(?,MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)		
        embedded_sequences = embedding_layer(X)
        #embedded_sequences_ex shape=(?,MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1)
        embedded_sequences_ex  = tf.expand_dims(embedded_sequences, -1)

    with tf.name_scope('first_conv_layer'):
    	# l1a shape=(?, 279, 1, 128)
        l1a = act(tf.nn.conv2d(embedded_sequences_ex, w,                                         
                                strides=[1, 1, 1, 1], padding='VALID'), 
                                name="first_activation")
        # l1  shape=(?, 279, 1, 128)
        l1 = tf.nn.max_pool(l1a, ksize=[1, MAX_SEQUENCE_LENGTH - 4 + 1, 1, 1],
                                strides=[1, 1, 1, 1], padding='VALID')
        l1 = tf.nn.dropout(l1, p_keep_conv)
        tf.summary.histogram('activations', l1)		
    
    with tf.name_scope('second_layer'):
    	# l2a shape=(?, 16, 16, 20)
        l2a = act(tf.nn.conv2d(l1, w2,                                                           
                                strides=[1, 1, 1, 1], padding='SAME'), 
                                name="second_activation")
        # l2  shape=(?, 8, 8, 20)
        l2 = tf.nn.max_pool(l2a, ksize=[1, 1, 1, 1],                                             
                                strides=[1, 1, 1, 1], padding='SAME')        
        l2 = tf.nn.dropout(l2, p_keep_conv)
        tf.summary.histogram('activations', l2)
	
    with tf.name_scope('third_layer'):
    	# l3a shape=(?, 8, 8, 20)
        l3a = act(tf.nn.conv2d(l2, w3,                                                           
                                strides=[1, 1, 1, 1], padding='SAME'), 
                                name="third_activation")
        # l3  shape=(?, 4, 4, 20)
        l3 = tf.nn.max_pool(l3a, ksize=[1, 1, 1, 1],                                             
                                strides=[1, 1, 1, 1], padding='SAME')
        l3 = tf.nn.dropout(l3, p_keep_conv)
        tf.summary.histogram('activations', l3)
		
    with tf.name_scope('output_layer'):
    	# reshape to (?, 4x4x20)
        l4 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])
        l4 = tf.nn.dropout(l4, p_keep_conv)

        l5 = act(tf.matmul(l4, w_fc))
        l5 = tf.nn.dropout(l5, p_keep_hidden)
    		
        pyx = tf.matmul(l5, w_o)
        print(l5)
        tf.summary.histogram('activations', l5)
        tf.summary.histogram('pyx', pyx)
        return pyx

# Use Keras to process the reviews and create the embedding layer
texts = []
labels = []
i = 0
f = open('train.tsv/train.tsv')
for line in f:
    if i > 0:
    	# Grab review and sentiment without the sentenceID or PhraseID
        lineArray = line.split()[2:]
        line_words = lineArray[:-1]
        line_label = lineArray[-1:]
        texts.append(line_words)
        labels.append(line_label)
    i += 1
words = []
for text in texts:
    word = ' '.join(text)
    words.append(word)
	
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(words)
sequences = tokenizer.texts_to_sequences(words)

word_index = tokenizer.word_index
print("Found {0} unique tokens.".format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))

# Split the data into training and testing while shuffling it randomly
# Testing data = 20% of training data
indices = np.arange(50000)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_testing = int(0.2 * data.shape[0])

trX = data[:-num_testing]
trY = labels[:-num_testing]
teX = data[-num_testing:]
teY = labels[-num_testing:]

with tf.name_scope('input'):
    X = tf.placeholder("float", [None, MAX_SEQUENCE_LENGTH])
    Y = tf.placeholder("float", [None, 5])

# Weight Matrices of the Convolutional Layers
with tf.name_scope('weights'):
    w = init_weights([4, EMBEDDING_DIM, 1, 128]) # conv, 16 outputs
    w2 = init_weights([4, 1, 128, 128])          # 5x5x16 conv, 20 outputs
    w3 = init_weights([4, 1, 128, 128])          # 5x5x20 conv, 20 outputs
    w_fc = init_weights([128, 5])                # FC 20 * 4 * 4 inputs, 10 outputs
    w_o = init_weights([5, 5])                   # FC 625 inputs, 10 outputs (labels)

with tf.name_scope('dropout'):	
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    tf.summary.scalar('dropout_keep_probability_convolutional', p_keep_conv)
    tf.summary.scalar('dropout_keep_probability_hidden', p_keep_hidden)

act = tf.nn.leaky_relu 

py_x = model(X, w, w2, w3, w_fc, w_o, p_keep_conv, p_keep_hidden, act)

with tf.name_scope('cost'): 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)

sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))

predict_op = tf.argmax(py_x, 1)

with tf.name_scope('accuracy'):
    acc = tf.placeholder("float")
tf.summary.scalar('accuracy', acc)

####################
# LAUNCH THE GRAPH IN A SESSION
####################
with tf.Session() as sess:
    # You need to initialize all variables
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('cnn/train', sess.graph)
    test_writer = tf.summary.FileWriter('cnn/test', sess.graph)
    tf.global_variables_initializer().run()

    for i in range(15):
        
        training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
        
        with tf.device('/gpu:0'):
            for start, end in training_batch:
                summary, _ = sess.run([merged, train_op], 
                	feed_dict={X: trX[start:end], Y: trY[start:end],
                    acc: 0.0, # Placeholder for recorded accuracy
                    p_keep_conv: 0.8, p_keep_hidden: 0.5}) 
        train_writer.add_summary(summary, i)

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        accuracy = np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, 
                        	feed_dict={X: teX[test_indices], Y: teY[test_indices],
                            acc:0.0, # Placeholder without value at first
                            p_keep_conv: 1.0,
                            p_keep_hidden: 1.0}))
        
        accuracy_t = tf.convert_to_tensor(accuracy, tf.float32)
                                                     
        with tf.name_scope('accuracy'):
            summary, accuracy = sess.run([merged, accuracy_t], 
            	feed_dict={X: teX[test_indices], Y: teY[test_indices],
                acc: accuracy, # Feed the accuracy to be graphed
                p_keep_conv: 1.0,                                              
                p_keep_hidden: 1.0})
        print(i, accuracy)
        test_writer.add_summary(summary, i)