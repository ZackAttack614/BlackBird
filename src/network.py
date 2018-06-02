import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Network:
    def __init__(self, saver, tfLog, loadOld=False, dims=(3,3), **kwargs):
        self.parameters = kwargs['network']
        self.dims = dims
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        self.createNetwork()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.batch_count = 0
        
        self.saver = tf.train.Saver()
        self.network_name = '{0}_{1}.ckpt'.format(self.parameters['blocks'], 
            self.parameters['filters'])
        self.model_loc = 'blackbird_models/best_model_{0}.ckpt'.format(self.network_name)
        self.writer_loc = 'blackbird_summary/model_summary'

        self.default_alpha = self.parameters['policy']['dirichlet']['alpha']
        self.default_epsilon = self.parameters['policy']['dirichlet']['epsilon']

        self.write_summary = tfLog
        
        if tfLog:
            self.writer = tf.summary.FileWriter(self.writer_loc, graph=self.sess.graph)
        
        if loadOld:
            try:
                self.loadModel()
            except:
                self.saveModel()

    def __del__(self):
        self.sess.close()
        try:
            self.writer.close()
        except:
            pass
    
    def createNetwork(self):
        """ Build out the policy/evaluation combo network
        """
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE) as scope:
            self.input = tf.placeholder(shape=[None, self.dims[0], self.dims[1], 3], name='board_input', dtype=tf.float32)
            self.correct_move_vec = tf.placeholder(shape=[None, self.dims[0] * self.dims[1]], name='correct_move_from_mcts', dtype=tf.float32)
            self.mcts_evaluation = tf.placeholder(shape=[None], name='mcts_evaluation', dtype=tf.float32)
            
        with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE) as scope:
            self.hidden = [self.input]

            
            with tf.variable_scope('conv_block', reuse=tf.AUTO_REUSE) as scope:
                self.hidden.append(
                    tf.layers.conv2d(inputs=self.input, filters=self.parameters['filters'], kernel_size=[3,3],
                        strides=1, padding="same", activation=None, name='conv'))
                self.hidden.append(tf.layers.batch_normalization(inputs=self.hidden[-1],name='batch_norm'))
                self.hidden.append(tf.nn.relu(features=self.hidden[-1], name='rectifier_nonlinearity'))
            
            for block in range(self.parameters['blocks']):
                with tf.variable_scope('block_{}'.format(block), reuse=tf.AUTO_REUSE) as scope:
                    self.hidden.append(
                        tf.layers.conv2d(inputs=self.hidden[-1], filters=self.parameters['filters'], kernel_size=[3,3],
                            strides=1, padding="same", activation=None, name='conv_1'))
                    self.hidden.append(tf.layers.batch_normalization(inputs=self.hidden[-1],name='batch_norm_1'))
                    self.hidden.append(tf.nn.relu(features=self.hidden[-1], name='rectifier_nonlinearity_1'))
                    self.hidden.append(
                        tf.layers.conv2d(inputs=self.hidden[-1], filters=self.parameters['filters'], kernel_size=[3,3],
                            strides=1, padding="same", activation=None, name='conv_2'))
                    self.hidden.append(tf.layers.batch_normalization(inputs=self.hidden[-1],name='batch_norm_2'))
                    self.hidden.append(tf.add(self.hidden[-1], self.hidden[-6], name='skip_connection'))
                    self.hidden.append(tf.nn.relu(features=self.hidden[-1], name='rectifier_nonlinearity_2'))
                    
        with tf.variable_scope('evaluation', reuse=tf.AUTO_REUSE) as scope:
            self.eval_conv = tf.layers.conv2d(self.hidden[-1],filters=1,kernel_size=(1,1),strides=1,name='convolution')
            self.eval_batch_norm = tf.layers.batch_normalization(self.eval_conv, name='batch_norm')
            self.eval_rectifier = tf.nn.relu(self.eval_batch_norm, name='rect_norm')
            self.eval_dense = tf.layers.dense(inputs=self.eval_rectifier, units=self.parameters['eval']['dense'], name='dense', activation=None)
            self.eval_scalar = tf.reduce_sum(self.eval_dense, axis=[1,2,3])
            self.evaluation = tf.tanh(self.eval_scalar, name='evaluation')
            
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE) as scope:
            self.epsilon = tf.placeholder(shape=[1], dtype=tf.float32)
            self.alpha = tf.placeholder(shape=[1], dtype=tf.float32)

            self.policy_conv = tf.layers.conv2d(self.hidden[-1],filters=2,kernel_size=(1,1),strides=1,name='convolution')
            self.policy_batch_norm = tf.layers.batch_normalization(self.policy_conv,name='batch_norm')
            self.policy_rectifier = tf.nn.relu(self.policy_batch_norm, name='rect_norm')
            self.policy_dense = tf.layers.dense(self.policy_rectifier, units=9, activation=None, name='policy')
            self.policy_vector = tf.reduce_sum(self.policy_dense, axis=[1,2])
            self.policy_base = tf.nn.softmax(self.policy_vector)

            self.dist = tf.distributions.Dirichlet([self.alpha[0], 1-self.alpha[0]])
            self.policy = (1-self.epsilon[0])*self.policy_base + self.epsilon[0] * self.dist.sample([1,9])[0][:,0]
            self.policy /= tf.reduce_sum(self.policy)
            
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as scope:
            self.loss_evaluation = tf.square(self.evaluation - self.mcts_evaluation)
            self.loss_policy = tf.reduce_sum(tf.tensordot( tf.log(self.policy), tf.transpose(self.correct_move_vec), axes=1), axis=1)
            self.loss_param = tf.tile(tf.expand_dims(tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                              if 'bias' not in v.name
                              ]) * self.parameters['loss']['L2_norm'], 0), [tf.shape(self.loss_policy)[0]])
            self.loss = tf.reduce_sum(self.loss_evaluation - self.loss_policy + self.loss_param)
            tf.summary.scalar('total_loss', self.loss)
            
        with tf.name_scope('summary') as scope:
            self.merged = tf.summary.merge_all()
            
        with tf.variable_scope('training', reuse=tf.AUTO_REUSE) as scope:
            self.learning_rate = tf.placeholder(shape=[1], dtype=tf.float32, name='learning_rate')

            if self.parameters['training']['optimizer'] == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate[0])
            elif self.parameters['training']['optimizer'] == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate[0], momentum=self.parameters['training']['momentum'])
            else:  # Default to SGD
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate[0])

            self.training_op = self.optimizer.minimize(self.loss)
            
    def getEvaluation(self, state):
        """ Given a game state, return the network's evaluation.
        """
        evaluation = self.sess.run(self.evaluation, feed_dict={self.input:state})
        return evaluation[0]
    
    def getPolicy(self, state):
        """ Given a game state, return the network's policy.
            Random Dirichlet noise is applied to the policy output to ensure exploration, if training.
        """
        policy = self.sess.run(self.policy, feed_dict={self.input:state, self.epsilon:[self.default_epsilon], self.alpha:[self.default_alpha]})
        return policy[0]
    
    def train(self, state, evaluation, policy, learning_rate=0.01):
        """ Train the network
        """
        feed_dict={
            self.input:state,
            self.mcts_evaluation:evaluation,
            self.correct_move_vec:policy,
            self.learning_rate:[learning_rate],
            self.epsilon:[self.default_epsilon],
            self.alpha:[self.default_alpha]
        }
        
        self.sess.run(self.training_op, feed_dict=feed_dict)
        self.batch_count += 1
        if self.batch_count % 10 == 0 and self.write_summary:
            summary = self.sess.run(self.merged, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.batch_count)
        
    def saveModel(self):
        """ Write the state of the network to a file.
            This should be reserved for "best" networks.
        """
        self.saver.save(self.sess, self.model_loc)
        
    def loadModel(self):
        """ Load an old version of the network.
        """
        self.saver.restore(self.sess, self.model_loc)
