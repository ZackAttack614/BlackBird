import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Network:
    def __init__(self, tfLog, dims, legalActions, teacher=False,
            loadOld=False, *args, **kwargs):

        self.parameters = kwargs
        self.dims = dims
        self.legalActions = legalActions

        gpuFrac = kwargs.get('gpu_frac')
        gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=gpuFrac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions))
        
        self.buildNetwork(teacher)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.batch_count = 0
        
        self.saver = tf.train.Saver()
        self.network_name = '{0}_{1}.ckpt'.format(self.parameters['blocks'], 
            self.parameters['filters'])

        self.model_loc = 'blackbird_models/best_model_{0}.ckpt'.format(
            self.network_name)
            
        self.writer_loc = 'blackbird_summary/model_summary'

        self.default_alpha = self.parameters.get('policy').get('dirichlet').get('alpha')
        self.default_epsilon = self.parameters.get('policy').get('dirichlet').get('epsilon')

        self.write_summary = tfLog
        
        if tfLog:
            self.writer = tf.summary.FileWriter(
                self.writer_loc, graph=self.sess.graph)
        
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
    
    def buildNetwork(self, hasTeacher):
        """ Build out the policy/evaluation combo network
        """
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE) as _:
            self.input = tf.placeholder(
                shape=[None] + self.dims + [3],
                name='board_input', dtype=tf.float32)

            self.correct_move_vec = tf.placeholder(
                shape=[None, self.legalActions],
                name='correct_move_from_mcts', dtype=tf.float32)

            self.mcts_evaluation = tf.placeholder(
                shape=[None], name='mcts_evaluation', dtype=tf.float32)
            
        with tf.variable_scope('res_tower', reuse=tf.AUTO_REUSE) as _:
            self.res_tower = [self.input]

            with tf.variable_scope('conv_block', reuse=tf.AUTO_REUSE) as _:
                """ AlphaZero convolutional blocks are...
                    1) Convolutional layer of 256 3x3 filters, stride of 1
                    2) Batch normalization
                    3) Rectifier nonlinearity
                """
                self.res_tower.append(
                    tf.layers.conv2d(inputs=self.input, kernel_size=[3,3],
                        strides=1, padding="same", name='conv',
                        filters=self.parameters['filters']))

                self.res_tower.append(
                    tf.layers.batch_normalization(
                        inputs=self.res_tower[-1],name='batch_norm'))
                
                self.res_tower.append(
                    tf.nn.relu(
                        features=self.res_tower[-1], name='rect_nonlinearity'))
            
            for block in range(self.parameters['blocks']):
                with tf.variable_scope('block_{}'.format(block), reuse=tf.AUTO_REUSE) as _:
                    """ AlphaZero residual blocks are...
                        1) Convolutional layer of 256 3x3 filters, stride of 1
                        2) Batch normalization
                        3) Rectifier nonlinearity
                        4) Convolutional layer of 256 3x3 filters, stride of 1
                        5) Bath normalization
                        6) Skip connection that adds the input to the block
                        7) Rectifier nonlinearity
                    """
                    self.res_tower.append(
                        tf.layers.conv2d(
                            inputs=self.res_tower[-1], kernel_size=[3,3], 
                            strides=1, filters=self.parameters['filters'],
                            padding="same", name='conv_1'))

                    self.res_tower.append(
                        tf.layers.batch_normalization(
                            inputs=self.res_tower[-1],name='batch_norm_1'))

                    self.res_tower.append(
                        tf.nn.relu(
                            features=self.res_tower[-1],
                            name='rectifier_nonlinearity_1'))

                    self.res_tower.append(
                        tf.layers.conv2d(
                            inputs=self.res_tower[-1], kernel_size=[3,3],
                            filters=self.parameters['filters'], strides=1,
                            padding="same", name='conv_2'))

                    self.res_tower.append(
                        tf.layers.batch_normalization(
                            inputs=self.res_tower[-1], name='batch_norm_2'))

                    self.res_tower.append(
                        tf.add(
                            self.res_tower[-1], self.res_tower[-6],
                            name='skip_connection'))

                    self.res_tower.append(
                        tf.nn.relu(
                            features=self.res_tower[-1],
                            name='rectifier_nonlinearity_2'))
                    
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE) as _:
            """ AlphaZero's value head is...
                1) Convolutional layer of 2 1x1 filters, stride of 1
                2) Batch normalization
                3) Rectifier nonlinearity
                4) Fully connected layer of size 256
                5) Rectifier nonlinearity
                6) Fully connected layer of size 1
                7) tanh activation
            """
            self.eval_conv = tf.layers.conv2d(
                self.res_tower[-1], filters=1, kernel_size=(1,1), strides=1,
                name='convolution')

            self.eval_batch_norm = tf.layers.batch_normalization(
                self.eval_conv, name='batch_norm')

            self.eval_rectifier_1 = tf.nn.relu(
                self.eval_batch_norm, name='rect_norm_1')

            self.eval_dense = tf.layers.dense(
                inputs=self.eval_rectifier_1,
                units=self.parameters.get('eval').get('dense'), name='dense_1')

            self.eval_dense_reduced = tf.reduce_sum(
                self.eval_dense,
                axis=[1,2], name='reduced_dense')

            self.eval_rectifier_2 = tf.nn.relu(
                self.eval_dense_reduced, name='rect_norm_2')

            self.eval_dense_scalar = tf.layers.dense(
                inputs=self.eval_rectifier_2,
                units=1, name='dense_2')

            self.eval_scalar = tf.reduce_sum(
                self.eval_dense_scalar,
                axis=[1], name='reduced_scalar')

            self.evaluation = tf.tanh(
                self.eval_scalar, name='value')
            
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE) as _:
            """ AlphaZero's policy head is...
                1) Convolutional layer of 2 1x1 filters, stride of 1
                2) Batch normalization
                3) Rectifier nonlinearity
                4) Fully connected layer of size |legal actions|
            """
            self.policy_conv = tf.layers.conv2d(
                self.res_tower[-1], filters=2, kernel_size=(1,1), strides=1,
                name='convolution')

            self.policy_batch_norm = tf.layers.batch_normalization(
                self.policy_conv, name='batch_norm')

            self.policy_rectifier = tf.nn.relu(
                self.policy_batch_norm, name='rect_norm')

            self.policy_dense = tf.layers.dense(
                self.policy_rectifier, units=self.legalActions, name='policy')

            self.policy_vector = tf.reduce_sum(
                self.policy_dense, axis=[1,2])

            self.policy_base = tf.nn.softmax(
                self.policy_vector)

            # Generate Dirichlet noise to add to the network policy
            self.epsilon = tf.placeholder(shape=[1], dtype=tf.float32)
            self.alpha = tf.placeholder(shape=[1], dtype=tf.float32)

            self.dist = tf.distributions.Dirichlet(
                [self.alpha[0], 1-self.alpha[0]])

            self.policy = ((1 - self.epsilon[0]) * self.policy_base
                + self.epsilon[0] * self.dist.sample([1, self.legalActions])[0][:,0])

            self.policy /= tf.reduce_sum(self.policy)
            
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as _:
            self.teacherPolicy = tf.placeholder(
                shape=[self.policy.shape[1]], dtype=tf.float32,
                name='teacher_policy')

            self.loss_evaluation = tf.reduce_mean(tf.square(
                self.evaluation - self.mcts_evaluation))

            self.loss_policy = -tf.reduce_mean(
                tf.tensordot(
                    tf.log(self.policy),
                    tf.transpose(self.correct_move_vec),
                    axes=1))

            self.loss_param = tf.reduce_mean([
                    tf.nn.l2_loss(v) for v in tf.trainable_variables()

                    # I don't know if this filter is a good idea...
                    if 'bias' not in v.name
                ])

            self.loss = self.loss_evaluation + self.loss_policy + self.loss_param

            if hasTeacher:
                self.policy_xentropy = -tf.reduce_mean(
                    tf.tensordot(
                        tf.log(self.teacherPolicy),
                        tf.transpose(self.policy),
                        axes=1),
                    axis=1)

                self.loss += self.policy_xentropy

            avg_loss = tf.summary.scalar('average_loss', self.loss)
            policy_loss = tf.summary.scalar('policy_loss', self.loss_policy)
            eval_loss = tf.summary.scalar('eval_loss', self.loss_evaluation)
            l2_loss = tf.summary.scalar('L2_Loss', self.loss_param)

            self.loss_merged = tf.summary.merge([avg_loss, policy_loss,
                                                 eval_loss, l2_loss])

        with tf.variable_scope('training', reuse=tf.AUTO_REUSE) as _:
            self.learning_rate = tf.placeholder(
                shape=[1], dtype=tf.float32, name='learning_rate')

            if self.parameters['training']['optimizer'] == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate[0])
            elif self.parameters['training']['optimizer'] == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(
                    self.learning_rate[0],
                    momentum=self.parameters['training']['momentum'])
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate[0])

            self.training_op = self.optimizer.minimize(self.loss)
            
    def getEvaluation(self, state):
        """ Given a game state, return the network's evaluation.
        """
        evaluation = self.sess.run(
            self.evaluation, feed_dict={self.input:state})
        
        return evaluation[0]
    
    def getPolicy(self, state):
        """ Given a game state, return the network's policy.
            Random Dirichlet noise is applied to the policy output
            to ensure exploration, if training.
        """
        policy = self.sess.run(
            self.policy, feed_dict={
                self.input:state,
                self.epsilon:[self.default_epsilon],
                self.alpha:[self.default_alpha]})

        return policy[0]
    
    def train(self, state, eval, policy, learning_rate=0.01, teacher=None):
        """ Train the network
        """
        feed_dict={
            self.input:state,
            self.mcts_evaluation:eval,
            self.correct_move_vec:policy,
            self.learning_rate:[learning_rate],
            self.epsilon:[self.default_epsilon],
            self.alpha:[self.default_alpha]
        }

        if teacher is not None:
            assert isinstance(teacher, Network), 'teacher can generate policies'
            feed_dict[self.teacherPolicy] = teacher.getPolicy(state)
        
        self.sess.run(self.training_op, feed_dict=feed_dict)
        self.batch_count += 1
        if self.batch_count % 10 == 0 and self.write_summary:
            summary = self.sess.run(self.loss_merged, feed_dict=feed_dict)
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
