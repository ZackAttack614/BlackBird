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
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.batchCount = 0

        self.saver = tf.train.Saver()
        self.network_name = '{0}_{1}.ckpt'.format(self.parameters['blocks'], 
            self.parameters['filters'])

        self.modelLoc = 'blackbird_models/best_model_{0}.ckpt'.format(
            self.network_name)

        self.writerLoc = 'blackbird_summary/model_summary'

        self.defaultAlpha = self.parameters.get('policy').get('dirichlet').get('alpha')
        self.defaultEpsilon = self.parameters.get('policy').get('dirichlet').get('epsilon')

        self.write_summary = tfLog

        if tfLog:
            self.writer = tf.summary.FileWriter(
                self.writerLoc, graph=self.sess.graph)

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

            self.mctsPolicy = tf.placeholder(
                shape=[None, self.legalActions],
                name='correct_move_from_mcts', dtype=tf.float32)

            self.mctsEvaluation = tf.placeholder(
                shape=[None], name='mctsEvaluation', dtype=tf.float32)

        with tf.variable_scope('resTower', reuse=tf.AUTO_REUSE) as _:
            self.resTower = [self.input]

            with tf.variable_scope('conv_block', reuse=tf.AUTO_REUSE) as _:
                """ AlphaZero convolutional blocks are...
                    1) Convolutional layer of 256 3x3 filters, stride of 1
                    2) Batch normalization
                    3) Rectifier nonlinearity
                """
                self.resTower.append(
                    tf.layers.conv2d(inputs=self.input, kernel_size=[3,3],
                        strides=1, padding="same", name='conv',
                        filters=self.parameters['filters']))

                self.resTower.append(
                    tf.layers.batch_normalization(
                        inputs=self.resTower[-1],name='batch_norm'))

                self.resTower.append(
                    tf.nn.relu(
                        features=self.resTower[-1], name='rect_nonlinearity'))

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
                    self.resTower.append(
                        tf.layers.conv2d(
                            inputs=self.resTower[-1], kernel_size=[3,3], 
                            strides=1, filters=self.parameters['filters'],
                            padding="same", name='conv_1'))

                    self.resTower.append(
                        tf.layers.batch_normalization(
                            inputs=self.resTower[-1],name='batch_norm_1'))

                    self.resTower.append(
                        tf.nn.relu(
                            features=self.resTower[-1],
                            name='rectifier_nonlinearity_1'))

                    self.resTower.append(
                        tf.layers.conv2d(
                            inputs=self.resTower[-1], kernel_size=[3,3],
                            filters=self.parameters['filters'], strides=1,
                            padding="same", name='conv_2'))

                    self.resTower.append(
                        tf.layers.batch_normalization(
                            inputs=self.resTower[-1], name='batch_norm_2'))

                    self.resTower.append(
                        tf.add(
                            self.resTower[-1], self.resTower[-6],
                            name='skip_connection'))

                    self.resTower.append(
                        tf.nn.relu(
                            features=self.resTower[-1],
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
            evalConv = tf.layers.conv2d(
                self.resTower[-1], filters=1, kernel_size=(1,1), strides=1,
                name='convolution')

            evalBatchNorm = tf.layers.batch_normalization(
                evalConv, name='batch_norm')

            evalRectifier1 = tf.nn.relu(
                evalBatchNorm, name='rect_norm_1')

            evalDense = tf.layers.dense(
                inputs=evalRectifier1,
                units=self.parameters.get('eval').get('dense'), name='dense_1')

            evalDenseReduced = tf.reduce_sum(
                evalDense,
                axis=[1,2], name='reduced_dense')

            evalRectifier2 = tf.nn.relu(
                evalDenseReduced, name='rect_norm_2')

            evalDenseScalar = tf.layers.dense(
                inputs=evalRectifier2,
                units=1, name='dense_2')

            evalScalar = tf.reduce_sum(
                evalDenseScalar,
                axis=[1], name='reduced_scalar')

            self.evaluation = tf.tanh(
                evalScalar, name='value')

        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE) as _:
            """ AlphaZero's policy head is...
                1) Convolutional layer of 2 1x1 filters, stride of 1
                2) Batch normalization
                3) Rectifier nonlinearity
                4) Fully connected layer of size |legal actions|
            """
            policyConv = tf.layers.conv2d(
                self.resTower[-1], filters=2, kernel_size=(1,1), strides=1,
                name='convolution')

            policyBatchNorm = tf.layers.batch_normalization(
                policyConv, name='batch_norm')

            policyRectifier = tf.nn.relu(
                policyBatchNorm, name='rect_norm')

            policyDense = tf.layers.dense(
                policyRectifier, units=self.legalActions, name='policy')

            policyVector = tf.reduce_sum(
                policyDense, axis=[1,2])

            policyBase = tf.nn.softmax(
                policyVector)

            # Generate Dirichlet noise to add to the network policy
            self.epsilon = tf.placeholder(shape=[1], dtype=tf.float32)
            self.alpha = tf.placeholder(shape=[1], dtype=tf.float32)

            dist = tf.distributions.Dirichlet(
                [self.alpha[0], 1-self.alpha[0]])

            self.policy = ((1 - self.epsilon[0]) * policyBase
                + self.epsilon[0] * dist.sample([1, self.legalActions])[0][:,0])

            self.policy /= tf.reduce_sum(self.policy)
            
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as _:
            teacherPolicy = tf.placeholder(
                shape=[self.policy.shape[1]], dtype=tf.float32,
                name='teacher_policy')

            lossEvaluation = tf.reduce_mean(tf.square(
                self.evaluation - self.mctsEvaluation))

            lossPolicy = -tf.reduce_mean(
                tf.tensordot(
                    tf.log(self.policy),
                    tf.transpose(self.mctsPolicy),
                    axes=1))

            lossParam = tf.reduce_mean([
                    tf.nn.l2_loss(v) for v in tf.trainable_variables()

                    # I don't know if this filter is a good idea...
                    if 'bias' not in v.name
                ])

            self.loss = lossEvaluation + lossPolicy + lossParam

            if hasTeacher:
                policyXentropy = -tf.reduce_mean(
                    tf.tensordot(
                        tf.log(self.teacherPolicy),
                        tf.transpose(self.policy),
                        axes=1),
                    axis=1)

                self.loss += policyXentropy

            avgLoss = tf.summary.scalar('average_loss', self.loss)
            policyLoss = tf.summary.scalar('policyLoss', lossPolicy)
            evalLoss = tf.summary.scalar('evalLoss', lossEvaluation)
            l2Loss = tf.summary.scalar('l2Loss', lossParam)

            self.lossMerged = tf.summary.merge([avgLoss, policyLoss,
                                                 evalLoss, l2Loss])

        with tf.variable_scope('training', reuse=tf.AUTO_REUSE) as _:
            self.learningRate = tf.placeholder(
                shape=[1], dtype=tf.float32, name='learningRate')

            if self.parameters['training']['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learningRate[0])
            elif self.parameters['training']['optimizer'] == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    self.learningRate[0],
                    momentum=self.parameters['training']['momentum'])
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    self.learningRate[0])

            self.trainingOp = optimizer.minimize(self.loss)

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
                self.epsilon:[self.defaultEpsilon],
                self.alpha:[self.defaultAlpha]})

        return policy[0]

    def train(self, state, eval, policy, learningRate=0.01, teacher=None):
        """ Train the network
        """
        feed_dict={
            self.input:state,
            self.mctsEvaluation:eval,
            self.mctsPolicy:policy,
            self.learningRate:[learningRate],
            self.epsilon:[self.defaultEpsilon],
            self.alpha:[self.defaultAlpha]
        }

        if teacher is not None:
            assert isinstance(teacher, Network), 'teacher can generate policies'
            feed_dict[self.teacherPolicy] = teacher.getPolicy(state)

        self.sess.run(self.trainingOp, feed_dict=feed_dict)
        self.batchCount += 1
        if self.batchCount % 10 == 0 and self.write_summary:
            summary = self.sess.run(self.lossMerged, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.batchCount)

    def saveModel(self):
        """ Write the state of the network to a file.
            This should be reserved for "best" networks.
        """
        self.saver.save(self.sess, self.modelLoc)

    def loadModel(self):
        """ Load an old version of the network.
        """
        self.saver.restore(self.sess, self.modelLoc)
