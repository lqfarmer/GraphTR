import tensorflow as tf

from .utils import *

class MultiAttentionAggregator(Layer):
    
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=True, num_heads=1, sample_num=1, **kwargs):
        super(MultiAttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.num_heads = num_heads

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')

            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')

            self.vars['inter_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='inter_weights')

            self.vars['inter_weights_mul'] = glorot([neigh_input_dim, output_dim],
                                                        name='inter_weights_mul')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')

            self.vars['output_weights'] = glorot([5 * output_dim, output_dim],
                                                        name='output_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='neigh_bias')


        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sample_num = sample_num
    
    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        #line part
        neigh_mean = tf.reduce_mean(neigh_vecs, axis=1)
        neigh_line = tf.matmul(neigh_mean, self.vars['neigh_weights'])

        #interaction_part
        pair_interactions = 0.5 *tf.subtract(
                tf.pow(
                    tf.matmul(neigh_mean,self.vars['inter_weights']),2),
                tf.matmul(tf.pow(neigh_mean,2),tf.pow(self.vars['inter_weights'],2)))

        # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
        query = tf.expand_dims(self_vecs, 1)
        neigh_self_vecs = tf.concat([neigh_vecs, query], axis=1)


        #attention multi
        neigh_self_vecs = split_heads(neigh_self_vecs, self.num_heads)
        query = split_heads(query, self.num_heads)
        logits = tf.matmul(query, neigh_self_vecs, transpose_b=True)
        score = tf.nn.softmax(logits, name="attention_weights")
        score = tf.nn.dropout(score, 1-self.dropout)
        #[batch_size,feature_size,node_nums,dims_fea]
        context = tf.matmul(score, neigh_self_vecs)
        context = combine_heads(context)
        context = tf.squeeze(context, [1])
        #interaction mul part
        pair_interactions_mul = 0.5 *tf.subtract(
                tf.pow(
                    tf.matmul(context,self.vars['inter_weights_mul']),2),
                tf.matmul(tf.pow(context,2),tf.pow(self.vars['inter_weights_mul'],2)))


        # [nodes] x [out_dim]
        from_neighs = tf.matmul(context, self.vars['weights'])

        if self.concat:
            # fully project
            output = tf.concat([from_self, from_neighs, neigh_line, pair_interactions, pair_interactions_mul], axis=1)
            output = tf.matmul(output, self.vars['output_weights'])
        else:
            # average project
            output = tf.add_n([from_self, from_neighs, neigh_line, pair_interactions, pair_interactions_mul])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu,
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):

        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
