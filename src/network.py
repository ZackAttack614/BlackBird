import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class network:
    def __init__(self, parameters, dims=(3,3), load_old=False):
        self.parameters = parameters
        self.dims = dims
        self.sess = tf.Session()
        
        self.createNetwork()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.batch_count = 0
        
        self.saver = tf.train.Saver()
        self.model_loc = 'blackbird_models/best_model.ckpt'
        self.writer_loc = 'blackbird_summary/model_summary'
        
        self.writer = tf.summary.FileWriter(self.writer_loc)
        
        if load_old:
            try:
                self.loadModel()
            except:
                self.saveModel()
    
    def createNetwork(self):
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE) as scope:
            self.input = tf.placeholder(shape=[1, self.dims[0], self.dims[1], 3], name='board_input', dtype=tf.float32)
            self.correct_move_vec = tf.placeholder(shape=[None], name='correct_move_from_mcts', dtype=tf.float32)
            self.mcts_evaluation = tf.placeholder(shape=[1], name='mcts_evaluation', dtype=tf.float32)
            
        with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE) as scope:
            self.hidden = [self.input]
            
            for block in range(self.parameters['blocks']):
                with tf.variable_scope('block_{}'.format(block), reuse=tf.AUTO_REUSE) as scope:
                    self.hidden.append(
                        tf.layers.conv2d(inputs=self.input, filters=self.parameters['filters'], kernel_size=[3,3],
                            strides=1, padding="same", activation=None, name='conv'))
                    self.hidden.append(tf.layers.batch_normalization(inputs=self.hidden[-1],name='batch_norm'))
                    self.hidden.append(tf.nn.relu(features=self.hidden[-1], name='rectifier_nonlinearity'))
                    
        with tf.variable_scope('evaluation', reuse=tf.AUTO_REUSE) as scope:
            self.eval_conv = tf.layers.conv2d(self.hidden[-1],filters=1,kernel_size=(1,1),strides=1,name='convolution')
            self.eval_batch_norm = tf.layers.batch_normalization(self.eval_conv, name='batch_norm')
            self.eval_rect_norm = tf.nn.relu(self.eval_batch_norm, name='rect_norm')
            self.eval_dense = tf.layers.dense(inputs=self.eval_rect_norm, units=self.parameters['eval']['dense'], name='dense', activation=tf.nn.relu)
            self.eval_slice = tf.slice(input_=self.eval_dense, begin=[0,0,0,0], size=[1,1,1,self.parameters['eval']['dense']])
            self.eval_scalar = tf.layers.dense(inputs=self.eval_slice, units=1, name='scalar')
            self.evaluation = tf.tanh(self.eval_scalar, name='evaluation')[0][0][0][0]
            
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE) as scope:
            self.policy_conv = tf.layers.conv2d(self.hidden[-1],filters=2,kernel_size=(1,1),strides=1,name='convolution')
            self.policy_batch_norm = tf.layers.batch_normalization(self.policy_conv,name='batch_norm')
            self.policy_rect_norm = tf.nn.relu(self.policy_batch_norm, name='rect_norm')
            self.policy = tf.layers.dense(self.policy_rect_norm, units=9, activation=tf.nn.softmax, name='policy')[0][0][0]
            
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as scope:
            self.loss_evaluation = tf.square(self.evaluation - self.mcts_evaluation)
            self.loss_policy = tf.tensordot(self.correct_move_vec, tf.log(self.policy), axes=1)
            self.loss_param = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                              #if 'bias' not in v.name
                              ]) * self.parameters['loss']['L2_norm']
            self.loss = self.loss_evaluation - self.loss_policy + self.loss_param
            tf.summary.scalar('total_loss', self.loss[0])
            
        with tf.name_scope('summary') as scope:
            self.merged = tf.summary.merge_all()
            
        with tf.variable_scope('training', reuse=tf.AUTO_REUSE) as scope:
            self.learning_rate = tf.placeholder(shape=[1], dtype=tf.float32, name='learning_rate')
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate[0])
            self.training_op = self.optimizer.minimize(self.loss)
            
    def getEvaluation(self, state):
        evaluation = self.sess.run(self.evaluation, feed_dict={self.input:state})
        return evaluation
    
    def getPolicy(self, state, noise=True, epsilon=0.25, alpha=0.03):
        policy = self.sess.run(self.policy, feed_dict={self.input:state})
        if True:
            while True:
                try:
                    policy = [(1-epsilon)*p + epsilon*np.random.dirichlet([alpha]) for p in policy]
                    break
                except:
                    continue
        return policy
    
    def train(self, state, evaluation, policy, learning_rate=0.01):
        feed_dict={
            self.input:state,
            self.mcts_evaluation:evaluation,
            self.correct_move_vec:policy,
            self.learning_rate:[learning_rate]
        }
        
        self.sess.run(self.training_op, feed_dict=feed_dict)
        self.batch_count += 1
        summary = self.sess.run(self.merged, feed_dict=feed_dict)
        self.writer.add_summary(summary, self.batch_count)
        
    def saveModel(self):
        self.saver.save(self.sess, self.model_loc)
        
    def loadModel(self):
        self.saver.restore(self.sess, self.model_loc)
