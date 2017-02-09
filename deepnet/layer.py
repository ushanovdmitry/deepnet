"""Implements a layer of neurons."""
import numpy as np

import cudamat as cm
import parameter


class ConvolutionOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.size = opts.pop("size", 0)  # optional int32 size = 1 [default=0];
        self.stride = opts.pop("stride", 0)  # optional int32 stride = 2 [default=0];
        self.padding = opts.pop("padding", 0)  # optional int32 padding = 3 [default=0];
        self.num_filters = opts.pop("num_filters", 0)  # optional int32 num_filters = 4 [default=0];
        self.num_colors = opts.pop("num_colors", 0)  # optional int32 num_colors = 5 [default=0];
        self.max_pool = opts.pop("max_pool", False)  # optional bool max_pool = 6 [default=false];
        self.pool_size = opts.pop("pool_size", 0)  # optional int32 pool_size = 7 [default=0];
        self.pool_stride = opts.pop("pool_stride", 0)  # optional int32 pool_stride = 8 [default=0];
        self.rnorm = opts.pop("rnorm", False)  # optional bool rnorm = 9 [default=false];
        self.norm_size = opts.pop("norm_size", 0)  # optional int32 norm_size = 10 [default=0];
        self.pow_scale = opts.pop("pow_scale", 0.75)  # optional float pow_scale = 11 [default=0.75];
        self.add_scale = opts.pop("add_scale", 0.001)  # optional float add_scale = 12 [default=0.001];
        self.prob = opts.pop("prob", False)  # optional bool prob = 13 [default=false];  probabilistic max_pool

        assert len(opts) == 0


class ParameterOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.name = opts.pop("name")  # required string name = 1;
        self.mat = opts.pop("mat", "")  # optional bytes mat = 2;
        self.dimensions = opts.pop("dimensions", [])  # repeated int32 dimensions = 3;
        self.initialization = opts.pop("initialization",
                                       "CONSTANT")  # optional Initialization initialization = 4 [default=CONSTANT];
        self.sigma = opts.pop("sigma", 0.001)  # optional float sigma = 5 [default=0.001];
        self.constant = opts.pop("constant", 0.0)  # optional float constant = 6 [default=0.0];
        self.conv = opts.pop("conv", False)  # optional bool conv = 7 [default=false];

        self.conv_params = ConvolutionOpts(opts.pop("conv_params", {}))  # optional Convolution conv_params = 8;
        self.pretrained_model = opts.pop("pretrained_model", [])  # repeated string pretrained_model = 9;
        self.pretrained_model_node1 = opts.pop("pretrained_model_node1",
                                               "")  # optional string pretrained_model_node1 = 10;
        self.pretrained_model_node2 = opts.pop("pretrained_model_node2",
                                               "")  # optional string pretrained_model_node2 = 11;
        self.transpose_pretrained = opts.pop("transpose_pretrained",
                                             False)  # optional bool transpose_pretrained = 12 [default=false];
        self.pretrained_model_param_name = opts.pop("pretrained_model_param_name",
                                                    "")  # optional string pretrained_model_param_name = 13;
        self.local = opts.pop("local", False)  # optional bool local = 14 [default=false];
        self.mult_factor = opts.pop("mult_factor", 1.0)  # optional float mult_factor = 15[default=1.0];

        assert len(opts) == 0


class HyperparamsOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.base_epsilon = opts.get("base_epsilon", 0.01)  #  optional float base_epsilon = 1 [default=0.01];
        self.epsilon_decay = opts.get("epsilon_decay", "NONE")  #  optional Decay epsilon_decay = 2 [default=NONE];
        self.epsilon_decay_half_life = opts.get("epsilon_decay_half_life", 1000)  #  optional int32 epsilon_decay_half_life = 3 [default=1000];
        self.initial_momentum = opts.get("initial_momentum", 0.0)  #  optional float initial_momentum = 4 [default=0.0];
        self.final_momentum = opts.get("final_momentum", 0.0)  #  optional float final_momentum = 5 [default=0.0];
        self.momentum_change_steps = opts.get("momentum_change_steps", 10)  #  optional int32 momentum_change_steps = 6 [default=10];
        self.sparsity = opts.get("sparsity", False)  #  optional bool sparsity = 7 [default=false];
        self.sparsity_target = opts.get("sparsity_target", 0.1)  #  optional float sparsity_target = 8 [default=0.1];
        self.sparsity_cost = opts.get("sparsity_cost", 0.001)  #  optional float sparsity_cost = 9 [default=0.001];
        self.sparsity_damping = opts.get("sparsity_damping", 0.9)  #  optional float sparsity_damping = 10 [default=0.9];
        self.dropout = opts.get("dropout", False)  #  optional bool dropout = 11 [default=false];
        self.dropout_prob = opts.get("dropout_prob", 0.5)  #  optional float dropout_prob = 12 [default=0.5];
        self.apply_weight_norm = opts.get("apply_weight_norm", False)  #  optional bool apply_weight_norm = 13 [default=false];
        self.weight_norm = opts.get("weight_norm", 10)  #  optional float weight_norm = 14 [default=10];
        self.apply_l2_decay = opts.get("apply_l2_decay", False)  #  optional bool apply_l2_decay = 15 [default=false];
        self.l2_decay = opts.get("l2_decay", 0.01)  #  optional float l2_decay = 16 [default=0.01];

        self.activation = opts.get("activation", "LINEAR")  #  optional Activation activation = 17 [default=LINEAR];
        self.left_window = opts.get("left_window", 0)  #  optional int32 left_window = 18 [default=0];
        self.right_window = opts.get("right_window", 0)  #  optional int32 right_window = 19 [default=0];
        self.mf_steps = opts.get("mf_steps", 1)  #  optional int32 mf_steps = 20 [default=1];
        self.gibbs_steps = opts.get("gibbs_steps", 1)  #  optional int32 gibbs_steps = 21 [default=1];

        self.adapt = opts.get("adapt", "NO_ADAPT")  #  optional Adapt adapt = 24 [default=NO_ADAPT];
        self.stop_dropout_for_last = opts.get("stop_dropout_for_last", 0)  #  optional int32 stop_dropout_for_last = 25 [default=0];
        self.enable_display = opts.get("enable_display", False)  #  optional bool enable_display = 26 [default=false];
        self.normalize = opts.get("normalize", False)  #  optional bool normalize = 27 [default=false];
        self.normalize_to = opts.get("normalize_to", 1.0)  #  optional float normalize_to = 28 [default=1.0];  // Replicated Softmax.
        self.start_learning_after = opts.get("start_learning_after", 0)  #  optional int32 start_learning_after = 29 [default=0];
        self.step_up_cd_after = opts.get("step_up_cd_after", 0)  #  optional int32 step_up_cd_after = 30 [default=0];
        self.decay_learning_rate_for_last = opts.get("decay_learning_rate_for_last", 0)  #  optional int32 decay_learning_rate_for_last = 31 [default=0];
        self.learn_precision = opts.get("learn_precision", False)  #  optional bool learn_precision = 32 [default=false];
        self.precision_epsilon = opts.get("precision_epsilon", 0.0)  #  optional float precision_epsilon = 33 [default=0.0];
        self.precision_upper_bound = opts.get("precision_upper_bound", 1.0)  #  optional float precision_upper_bound = 34 [default=1.0];
        self.apply_l1_decay = opts.get("apply_l1_decay", False)  #  optional bool apply_l1_decay = 35 [default=false];
        self.l1_decay = opts.get("l1_decay", 0.01)  #  optional float l1_decay = 36 [default=0.01];
        self.apply_l1decay_after = opts.get("apply_l1decay_after", 0)  #  optional int32 apply_l1decay_after = 37 [default=0];
        self.add_noise = opts.get("add_noise", False)  #  optional bool add_noise = 38 [default=false];
        self.shift = opts.get("shift", False)  #  optional bool shift = 39 [default=false];
        self.shift_amt_x = opts.get("shift_amt_x", 0)  #  optional int32 shift_amt_x = 40 [default=0];
        self.blocksize = opts.get("blocksize", 1)  #  optional int32 blocksize = 41 [default=1];  // block size for group dropout
        self.shift_amt_y = opts.get("shift_amt_y", 0)  #  optional int32 shift_amt_y = 42 [default=0];
        self.sc_alpha = opts.get("sc_alpha", 0)  #  optional float sc_alpha = 43 [default=0]; // alpha in Predictive Sparse Decomp.
        self.sc_beta = opts.get("sc_beta", 0)  #  optional float sc_beta = 44 [default=0]; // beta in Predictive Sparse Decomp.
        self.sc_gamma = opts.get("sc_gamma", 0)  #  optional float sc_gamma = 45 [default=0]; // gamma in PSD.
        self.sample_input = opts.get("sample_input", False)  #  optional bool sample_input = 46 [default=false];  // sample inputs when training RBMs.
        self.mult_dropout = opts.get("mult_dropout", False)  #  optional bool mult_dropout = 47 [default=false];  // Multiplicative gaussian dropout.
        self.select_model_using_error = opts.get("select_model_using_error", False)  #  optional bool select_model_using_error = 48 [default=false];

        # In DBMs/RBMs start sampling inputs after these many steps (if sample_input is true).
        self.sample_input_after = opts.get("sample_input_after", 0)  #  optional int32 sample_input_after = 49 [default=0];

        # Replicated softmax models with replicated priors.
        self.additive_prior = opts.get("additive_prior", 0)  #  optional int32 additive_prior = 50 [default=0];
        self.multiplicative_prior = opts.get("multiplicative_prior", 0)  #  optional int32 multiplicative_prior = 51 [default=0];
        self.adaptive_prior = opts.get("adaptive_prior", 0)  #  optional int32 adaptive_prior = 52 [default=0];
        self.normalize_error = opts.get("normalize_error", False)  #  optional bool normalize_error = 53 [default=false];

        self.start_step_up_cd_after = opts.get("start_step_up_cd_after", 0)  #  optional int32 start_step_up_cd_after = 54 [default=0];
        self.select_model_using_acc = opts.get("select_model_using_acc", False)  #  optional bool select_model_using_acc = 55 [default=false];
        self.select_model_using_map = opts.get("select_model_using_map", False)  #  optional bool select_model_using_map = 56 [default=false];

        self.fast_dropout = opts.get("fast_dropout", False)  #  optional bool fast_dropout = 57 [default=false];
        self.fast_dropout_cost = opts.get("fast_dropout_cost", 0)  #  optional float fast_dropout_cost = 58 [default=0];

        self.shared_prior = opts.get("shared_prior", False)  #  optional bool shared_prior = 59 [default=false];
        self.shared_prior_file = opts.get("shared_prior_file", "")  #  optional string shared_prior_file = 60;
        self.shared_prior_edge = opts.get("shared_prior_edge", "")  #  optional string shared_prior_edge = 61;
        self.shared_prior_cost = opts.get("shared_prior_cost", 0.0)  #  optional float shared_prior_cost = 62 [default=0.0];
        self.soft_shared_prior = opts.get("soft_shared_prior", False)  #  optional bool soft_shared_prior = 63 [default=false];
        self.label_freq_file = opts.get("label_freq_file", "")  #  optional string label_freq_file = 64;

        assert len(opts) == 0


class DataFieldOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.train = opts.pop("train", "")  # optional string train = 1;
        self.validation = opts.pop("validation", "")  # optional string validation = 2;
        self.test = opts.pop("test", "")  # optional string test = 3;

        # Data will be forward passed through this model and used as input.
        self.model = opts.pop("model", "")  # optional string model = 4;
        # The layer in the above model whose data is going to be the input.
        self.layer_name = opts.pop("layer_name", "")  # optional string layer_name = 5;
        self.tied = opts.pop("tied", False)  # optional bool tied = 6 [default=false];
        self.tied_to = opts.pop("tied_to", "")  # optional string tied_to = 7;

        assert len(opts) == 0


class MetricsOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)


class LayerOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.name = opts.pop("name")
        self.file_pattern = opts.pop("file_pattern", "")
        self.dimensions = opts.pop("dimensions")
        self.numlabels = opts.pop("numlabels", 1)
        self.param = [ParameterOpts(_) for _ in opts.pop("param", [])]  # repeated deepnet.Parameter param = 5;
        self.is_input = opts.pop("is_input", False)  # optional bool is_input = 7 [default=false];
        self.is_output = opts.pop("is_output", False)  # optional bool is_output = 8 [default=false];

        self.loss_function = opts.pop("loss_function",
                                      "SQUARED_LOSS")  # optional LossFunction loss_function = 9 [default=SQUARED_LOSS];
        self.hyperparams = HyperparamsOpts(
            opts.pop("hyperparams", {}))  # optional deepnet.Hyperparams hyperparams = 10;

        self.data_field = DataFieldOpts(opts.pop("data_field", {}))  # optional DataField data_field = 11;
        self.performance_stats = MetricsOpts(
            opts.pop("performance_stats", {}))  # optional deepnet.Metrics performance_stats = 12;
        self.shape = opts.pop("shape", [])  # repeated int32 shape = 13;
        self.is_initialized = opts.pop("is_initialized", False)  # optional bool is_initialized = 14 [default=false];
        self.prefix = opts.pop("prefix", "")  # optional string prefix = 15;
        self.replicate_bias = opts.pop("replicate_bias", True)  # optional bool replicate_bias = 16 [default=true];

        # Tying bias parameter.
        self.tied = opts.pop("tied", False)  # optional bool tied = 17 [default=false];
        self.tied_to = opts.pop("tied_to", "")  # optional string tied_to = 18;

        self.loss_weight = opts.pop("loss_weight", 1.0)  # optional float loss_weight = 19 [default=1.0];

        assert len(opts) == 0


class Layer(parameter.Parameter):
    def __init__(self, opts, t_op=None, tied_to=None):
        super(Layer, self).__init__()
        self.tied_to = tied_to
        if opts.get("tied", False):
            tied_to.num_shares += 1
            proto = util.LoadMissing(proto, tied_to.proto)
        self.opts = opts
        self.state = None
        self.params = {}
        self.hyperparams = opts["hyperparams"]
        self.incoming_edge = []
        self.outgoing_edge = []
        self.outgoing_neighbour = []
        self.incoming_neighbour = []
        self.use_suff_stats = False
        self.fast_dropout_partner = None
        if t_op:
            self.batchsize = t_op["batchsize"]
            self.use_suff_stats = t_op["optimizer"] in ["PCD", "CD"]
        else:
            self.batchsize = 0
        self.name = opts["name"]
        self.dimensions = opts["dimensions"]
        self.numlabels = opts.get("numlabels", 1)
        self.activation = opts["hyperparams"]["activation"]
        self.is_input = opts["is_input"]
        self.is_output = opts["is_output"]
        self.loss_function = opts["loss_function"]
        self.loss_weight = opts["loss_weight"]
        self.train_data_handler = None
        self.validation_data_handler = None
        self.test_data_handler = None
        self.tied_to = None
        self.data_tied_to = None
        self.data = None
        self.deriv = None
        self.prefix = opts.get("prefix", "")
        self.marker = 0
        self.tiny = 1e-10
        self.replicated_neighbour = None
        self.is_initialized = opts.get("is_initialized", False)
        self.t_op = t_op
        self.learn_precision = False
        self.sample_input = self.hyperparams["sample_input"]
        self.LoadParams(proto, t_op=t_op, tied_to=tied_to)
        if self.batchsize > 0:
            self.AllocateMemory(self.batchsize)

    def LoadParams(self, proto, **kwargs):
        assert proto
        for param in proto.param:
            if not param.dimensions:
                param.dimensions.extend([proto.numlabels * proto.dimensions, 1])
            elif len(param.dimensions) == 1:
                param.dimensions.append(1)
        super(Layer, self).LoadParams(proto, **kwargs)

    def LoadPretrained(self, param):
        node_name = param.pretrained_model_node1
        if node_name == '':
            node_name = self.proto.name
        mat = None
        for pretrained_model in param.pretrained_model:
            model_file = os.path.join(self.prefix, pretrained_model)
            ext = os.path.splitext(pretrained_model)[1]
            if ext == '.npz':
                npzfile = np.load(model_file)
                if param.name == 'bias':
                    this_mat = np.nan_to_num(npzfile['mean'] / npzfile['std'])
                elif param.name == 'precision':
                    this_mat = np.nan_to_num(1. / npzfile['std'])
            elif ext == '.npy':
                this_mat = np.load(model_file)
            else:
                model = util.ReadModel(model_file)
                # Find the relevant node in the model.
                node = next(n for n in model.layer if n.name == node_name)
                # Find the relevant parameter in the node.
                pretrained_param = next(p for p in node.param if p.name == param.name)
                assert pretrained_param.mat != '', \
                    'Pretrained param %s in layer %s of model %s is empty!!' % (
                        pretrained_param.name, node.name, pretrained_model)
                this_mat = util.ParameterAsNumpy(pretrained_param)
            if len(this_mat.shape) == 1:
                this_mat = this_mat.reshape(-1, 1)
            if mat is None:
                mat = this_mat
            else:
                mat += this_mat
        return mat / len(param.pretrained_model)

    def SetData(self, data):
        self.data = data

    def AddIncomingEdge(self, edge):
        if edge not in self.incoming_edge:
            self.incoming_edge.append(edge)
            if self == edge.node1:
                neighbour = edge.node2
            else:
                neighbour = edge.node1
            self.incoming_neighbour.append(neighbour)
            if neighbour.proto.replicate_bias and neighbour.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
                self.replicated_neighbour = neighbour

    def AddOutgoingEdge(self, edge):
        if edge not in self.outgoing_edge:
            self.outgoing_edge.append(edge)
            if self == edge.node1:
                self.outgoing_neighbour.append(edge.node2)
            else:
                self.outgoing_neighbour.append(edge.node1)

    def PrintNeighbours(self):
        for n in self.incoming_neighbour:
            print "Incoming edge from %s" % n.name
        for n in self.outgoing_neighbour:
            print "Outgoing edge to %s" % n.name

    def ResetState(self, rand=False):
        if rand:
            self.state.fill_with_randn()
            self.ApplyActivation()
        else:
            self.state.assign(0)

    def GetData(self):
        self.state.assign(self.data)

    def GetSparsityGradient(self):
        h = self.hyperparams
        damping = h.sparsity_damping
        target = h.sparsity_target
        cost = h.sparsity_cost

        # Update \hat{\rho}.
        self.means.mult(damping)
        self.means.add_sums(self.state, axis=1, mult=(1 - damping) / self.batchsize)

        # Compute gradient.
        self.means.subtract(target, target=self.sparsity_gradient)
        div = self.GetSparsityDivisor()
        self.sparsity_gradient.divide(div)
        self.sparsity_gradient.mult(cost)

        # Return gradient.
        return self.sparsity_gradient

    def AllocateMemory(self, batchsize):
        self.AllocateBatchsizeDependentMemory(batchsize)
        dimensions = self.dimensions
        numlabels = self.numlabels
        numdims = dimensions * numlabels
        self.dimsize = cm.CUDAMatrix(np.zeros((numdims, 1)))
        if self.hyperparams.sparsity:
            tgt = self.hyperparams.sparsity_target
            self.means = cm.CUDAMatrix(tgt + np.zeros((numdims, 1)))
            self.sparsity_gradient = cm.CUDAMatrix(np.zeros((numdims, 1)))
            self.means_temp2 = cm.CUDAMatrix(np.zeros((numdims, 1)))
        self.gradient = cm.CUDAMatrix(np.zeros((numdims, 1)))
        self.gradient_history = cm.CUDAMatrix(np.zeros((numdims, 1)))

    def AllocateBatchsizeDependentMemory(self, batchsize):
        if self.data:
            self.data.free_device_memory()
        if self.deriv:
            self.deriv.free_device_memory()
        self.batchsize = batchsize
        dimensions = self.dimensions
        numlabels = self.numlabels
        numdims = dimensions * numlabels
        self.statesize = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.batchsize_temp = cm.CUDAMatrix(np.zeros((1, batchsize)))
        self.state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.deriv = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        if self.t_op:
            if self.t_op.optimizer == deepnet_pb2.Operation.PCD:
                self.pos_state = self.state
                self.pos_sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
                self.neg_state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
                self.neg_sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
                self.sample = self.pos_sample
                self.suff_stats = cm.empty((numdims, 1))
            elif self.t_op.optimizer == deepnet_pb2.Operation.CD:
                self.sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
                self.suff_stats = cm.empty((numdims, 1))
        else:
            self.state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        if self.is_input or self.is_initialized or self.is_output:
            self.data = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
        if self.hyperparams.dropout:
            self.mask = cm.CUDAMatrix(np.zeros(self.state.shape))

    def CollectSufficientStatistics(self, neg=False):
        """Collect sufficient statistics for this layer."""
        h = self.hyperparams
        if not neg:
            self.state.sum(axis=1, target=self.suff_stats)
            if h.sparsity:
                sparsity_gradient = self.GetSparsityGradient()
                self.suff_stats.add_mult(sparsity_gradient, -self.batchsize)
        else:
            self.suff_stats.add_sums(self.state, axis=1, mult=-1.0)
        if not neg and h.sparsity:
            return self.means.sum() / self.means.shape[0]

    def Show(self, train=False):
        """Displays useful statistics about the model."""
        if not self.proto.hyperparams.enable_display:
            return
        f = 1
        if self.hyperparams.dropout and not train:
            f = 1 / (1 - self.hyperparams.dropout_prob)
        if self.is_input:
            visualize.display_hidden(self.data.asarray(), self.fig, title=self.name)
            # visualize.display_w(self.neg_sample.asarray(), 28, 10, self.state.shape[1]/10, self.fig, title=self.name, vmax=1, vmin=0)
            # visualize.show_hist(self.params['bias'].asarray(), self.fig)
        else:
            visualize.display_hidden(f * self.state.asarray(), self.fig, title=self.name)
            # visualize.show_hist(self.params['bias'].asarray(), self.fig)
            """
            plt.figure(self.fig)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.title('pos_probabilities')
            plt.imshow(self.pos_state.asarray(), cmap = plt.cm.gray, interpolation = 'nearest', vmax=1, vmin=0)
            plt.subplot(1, 3, 2)
            plt.title('neg_probabilities')
            plt.imshow(self.neg_state.asarray(), cmap = plt.cm.gray, interpolation = 'nearest', vmax=1, vmin=0)
            plt.subplot(1, 3, 3)
            plt.title('neg_samples')
            plt.imshow(self.neg_sample.asarray(), cmap = plt.cm.gray, interpolation = 'nearest', vmax=1, vmin=0)
            plt.suptitle(self.name)
            plt.draw()
            """
            # visualize.display_w(self.neg_sample.asarray(), 1, 1, self.state.shape[1], self.fig, title=self.name)


def display_w(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist', title='weights'):
    def ComputeDeriv(self):
        pass

    def GetLoss(self, get_deriv=False):
        pass

    def Sample(self):
        pass

    def ApplyActivation(self):
        pass

    def GetSparsityDivisor(self):
        self.means_temp2.assign(1)
        return self.means_temp2
