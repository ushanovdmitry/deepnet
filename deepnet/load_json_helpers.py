class ConvolutionOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.size = opts.pop("size", 0)  # optional int32 size = 1 [default=0];
        self.stride = opts.pop("stride", 0)  # optional int32 stride = 2 [default=0];
        self.padding = opts.pop("padding", 0)  # optional int32 padding = 3 [default=0];
        self.num_filters = opts.pop("numFilters", 0)  # optional int32 num_filters = 4 [default=0];
        self.num_colors = opts.pop("numColors", 0)  # optional int32 num_colors = 5 [default=0];
        self.max_pool = opts.pop("maxPool", False)  # optional bool max_pool = 6 [default=false];
        self.pool_size = opts.pop("poolSize", 0)  # optional int32 pool_size = 7 [default=0];
        self.pool_stride = opts.pop("poolStride", 0)  # optional int32 pool_stride = 8 [default=0];
        self.rnorm = opts.pop("rnorm", False)  # optional bool rnorm = 9 [default=false];
        self.norm_size = opts.pop("normSize", 0)  # optional int32 norm_size = 10 [default=0];
        self.pow_scale = opts.pop("powScale", 0.75)  # optional float pow_scale = 11 [default=0.75];
        self.add_scale = opts.pop("addScale", 0.001)  # optional float add_scale = 12 [default=0.001];
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

        self.conv_params = ConvolutionOpts(opts.pop("convParams", {}))  # optional Convolution conv_params = 8;
        self.pretrained_model = opts.pop("pretrainedModel", [])  # repeated string pretrained_model = 9;
        self.pretrained_model_node1 = opts.pop("pretrainedModelNode1",
                                               "")  # optional string pretrained_model_node1 = 10;
        self.pretrained_model_node2 = opts.pop("pretrainedModelNode2",
                                               "")  # optional string pretrained_model_node2 = 11;
        self.transpose_pretrained = opts.pop("transposePretrained",
                                             False)  # optional bool transpose_pretrained = 12 [default=false];
        self.pretrained_model_param_name = opts.pop("pretrainedModelParamName",
                                                    "")  # optional string pretrained_model_param_name = 13;
        self.local = opts.pop("local", False)  # optional bool local = 14 [default=false];
        self.mult_factor = opts.pop("multFactor", 1.0)  # optional float mult_factor = 15[default=1.0];

        assert len(opts) == 0


class HyperparamsOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.base_epsilon = opts.pop("baseEpsilon", 0.01)  # optional float base_epsilon = 1 [default=0.01];
        self.epsilon_decay = opts.pop("epsilonDecay", "NONE")  # optional Decay epsilon_decay = 2 [default=NONE];
        self.epsilon_decay_half_life = opts.pop("epsilonDecayHalfLife",
                                                1000)  # optional int32 epsilon_decay_half_life = 3 [default=1000];
        self.initial_momentum = opts.pop("initialMomentum", 0.0)  # optional float initial_momentum = 4 [default=0.0];
        self.final_momentum = opts.pop("finalMomentum", 0.0)  # optional float final_momentum = 5 [default=0.0];
        self.momentum_change_steps = opts.pop("momentumChangeSteps",
                                              10)  # optional int32 momentum_change_steps = 6 [default=10];
        self.sparsity = opts.pop("sparsity", False)  # optional bool sparsity = 7 [default=false];
        self.sparsity_target = opts.pop("sparsityTarget", 0.1)  # optional float sparsity_target = 8 [default=0.1];
        self.sparsity_cost = opts.pop("sparsityCost", 0.001)  # optional float sparsity_cost = 9 [default=0.001];
        self.sparsity_damping = opts.pop("sparsityDamping", 0.9)  # optional float sparsity_damping = 10 [default=0.9];
        self.dropout = opts.pop("dropout", False)  # optional bool dropout = 11 [default=false];
        self.dropout_prob = opts.pop("dropoutProb", 0.5)  # optional float dropout_prob = 12 [default=0.5];
        self.apply_weight_norm = opts.pop("applyWeightNorm",
                                          False)  # optional bool apply_weight_norm = 13 [default=false];
        self.weight_norm = opts.pop("weightNorm", 10)  # optional float weight_norm = 14 [default=10];
        self.apply_l2_decay = opts.pop("applyL2Decay", False)  # optional bool apply_l2_decay = 15 [default=false];
        self.l2_decay = opts.pop("l2Decay", 0.01)  # optional float l2_decay = 16 [default=0.01];

        self.activation = opts.pop("activation", "LINEAR")  # optional Activation activation = 17 [default=LINEAR];
        self.left_window = opts.pop("leftWindow", 0)  # optional int32 left_window = 18 [default=0];
        self.right_window = opts.pop("rightWindow", 0)  # optional int32 right_window = 19 [default=0];
        self.mf_steps = opts.pop("mfSteps", 1)  # optional int32 mf_steps = 20 [default=1];
        self.gibbs_steps = opts.pop("gibbsSteps", 1)  # optional int32 gibbs_steps = 21 [default=1];

        self.adapt = opts.pop("adapt", "NO_ADAPT")  # optional Adapt adapt = 24 [default=NO_ADAPT];
        self.stop_dropout_for_last = opts.pop("stopDropoutForLast",
                                              0)  # optional int32 stop_dropout_for_last = 25 [default=0];
        self.enable_display = opts.pop("enableDisplay", False)  # optional bool enable_display = 26 [default=false];
        self.normalize = opts.pop("normalize", False)  # optional bool normalize = 27 [default=false];
        self.normalize_to = opts.pop("normalizeTo",
                                     1.0)  # optional float normalize_to = 28 [default=1.0];  // Replicated Softmax.
        self.start_learning_after = opts.pop("startLearningAfter",
                                             0)  # optional int32 start_learning_after = 29 [default=0];
        self.step_up_cd_after = opts.pop("stepUpCdAfter", 0)  # optional int32 step_up_cd_after = 30 [default=0];
        self.decay_learning_rate_for_last = opts.pop("decayLearningRateForLast",
                                                     0)  # optional int32 decay_learning_rate_for_last = 31 [default=0];
        self.learn_precision = opts.pop("learnPrecision", False)  # optional bool learn_precision = 32 [default=false];
        self.precision_epsilon = opts.pop("precisionEpsilon",
                                          0.0)  # optional float precision_epsilon = 33 [default=0.0];
        self.precision_upper_bound = opts.pop("precisionUpperBound",
                                              1.0)  # optional float precision_upper_bound = 34 [default=1.0];
        self.apply_l1_decay = opts.pop("applyL1Decay", False)  # optional bool apply_l1_decay = 35 [default=false];
        self.l1_decay = opts.pop("l1Decay", 0.01)  # optional float l1_decay = 36 [default=0.01];
        self.apply_l1decay_after = opts.pop("applyL1decayAfter",
                                            0)  # optional int32 apply_l1decay_after = 37 [default=0];
        self.add_noise = opts.pop("addNoise", False)  # optional bool add_noise = 38 [default=false];
        self.shift = opts.pop("shift", False)  # optional bool shift = 39 [default=false];
        self.shift_amt_x = opts.pop("shiftAmtX", 0)  # optional int32 shift_amt_x = 40 [default=0];
        self.blocksize = opts.pop("blocksize",
                                  1)  # optional int32 blocksize = 41 [default=1];  // block size for group dropout
        self.shift_amt_y = opts.pop("shiftAmtY", 0)  # optional int32 shift_amt_y = 42 [default=0];
        self.sc_alpha = opts.pop("scAlpha",
                                 0)  # optional float sc_alpha = 43 [default=0]; // alpha in Predictive Sparse Decomp.
        self.sc_beta = opts.pop("scBeta",
                                0)  # optional float sc_beta = 44 [default=0]; // beta in Predictive Sparse Decomp.
        self.sc_gamma = opts.pop("scGamma", 0)  # optional float sc_gamma = 45 [default=0]; // gamma in PSD.
        self.sample_input = opts.pop("sampleInput",
                                     False)  # optional bool sample_input = 46 [default=false];  // sample inputs when training RBMs.
        self.mult_dropout = opts.pop("multDropout",
                                     False)  # optional bool mult_dropout = 47 [default=false];  // Multiplicative gaussian dropout.
        self.select_model_using_error = opts.pop("selectModelUsingError",
                                                 False)  # optional bool select_model_using_error = 48 [default=false];

        # In DBMs/RBMs start sampling inputs after these many steps (if sample_input is true).
        self.sample_input_after = opts.pop("sampleInputAfter",
                                           0)  # optional int32 sample_input_after = 49 [default=0];

        # Replicated softmax models with replicated priors.
        self.additive_prior = opts.pop("additivePrior", 0)  # optional int32 additive_prior = 50 [default=0];
        self.multiplicative_prior = opts.pop("multiplicativePrior",
                                             0)  # optional int32 multiplicative_prior = 51 [default=0];
        self.adaptive_prior = opts.pop("adaptivePrior", 0)  # optional int32 adaptive_prior = 52 [default=0];
        self.normalize_error = opts.pop("normalizeError", False)  # optional bool normalize_error = 53 [default=false];

        self.start_step_up_cd_after = opts.pop("startStepUpCdAfter",
                                               0)  # optional int32 start_step_up_cd_after = 54 [default=0];
        self.select_model_using_acc = opts.pop("selectModelUsingAcc",
                                               False)  # optional bool select_model_using_acc = 55 [default=false];
        self.select_model_using_map = opts.pop("selectModelUsingMap",
                                               False)  # optional bool select_model_using_map = 56 [default=false];

        self.fast_dropout = opts.pop("fastDropout", False)  # optional bool fast_dropout = 57 [default=false];
        self.fast_dropout_cost = opts.pop("fastDropoutCost", 0)  # optional float fast_dropout_cost = 58 [default=0];

        self.shared_prior = opts.pop("sharedPrior", False)  # optional bool shared_prior = 59 [default=false];
        self.shared_prior_file = opts.pop("sharedPriorFile", "")  # optional string shared_prior_file = 60;
        self.shared_prior_edge = opts.pop("sharedPriorEdge", "")  # optional string shared_prior_edge = 61;
        self.shared_prior_cost = opts.pop("sharedPriorCost",
                                          0.0)  # optional float shared_prior_cost = 62 [default=0.0];
        self.soft_shared_prior = opts.pop("softSharedPrior",
                                          False)  # optional bool soft_shared_prior = 63 [default=false];
        self.label_freq_file = opts.pop("labelFreqFile", "")  # optional string label_freq_file = 64;

        assert len(opts) == 0, opts


class DataFieldOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.train = opts.pop("train", "")  # optional string train = 1;
        self.validation = opts.pop("validation", "")  # optional string validation = 2;
        self.test = opts.pop("test", "")  # optional string test = 3;

        # Data will be forward passed through this model and used as input.
        self.model = opts.pop("model", "")  # optional string model = 4;
        # The layer in the above model whose data is going to be the input.
        self.layer_name = opts.pop("layerName", "")  # optional string layer_name = 5;
        self.tied = opts.pop("tied", False)  # optional bool tied = 6 [default=false];
        self.tied_to = opts.pop("tiedTo", "")  # optional string tied_to = 7;

        assert len(opts) == 0


class MetricsOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.count = opts.pop("count", 0)  #  optional int32 count = 1 [default=0];
        self.correct_preds = opts.pop("correctPreds", 0.0)  #  optional float correct_preds = 2;
        self.compute_correct_preds = opts.pop("computeCorrectPreds", False)  #  optional bool compute_correct_preds = 3 [default=false];
        self.cross_entropy = opts.pop("crossEntropy", 0.0)  #  optional float cross_entropy = 4;
        self.compute_cross_entropy = opts.pop("computeCrossEntropy", False)  #  optional bool compute_cross_entropy = 5 [default=false];
        self.error = opts.pop("error", 0.0)  #  optional float error = 6;
        self.compute_error = opts.pop("computeError", False)  #  optional bool compute_error = 7 [default=false];
        self.MAP = opts.pop("MAP", 0.0)  #  optional float MAP = 8;
        self.compute_MAP = opts.pop("computeMAP", False)  #  optional bool compute_MAP = 9 [default=false];
        self.prec50 = opts.pop("prec50", 0.0)  #  optional float prec50 = 10;
        self.compute_prec50 = opts.pop("computePrec50", False)  #  optional bool compute_prec50 = 11 [default=false];
        self.MAP_list = opts.pop("MAPList", 0.0)  #  repeated float MAP_list = 12;
        self.prec50_list = opts.pop("prec50List", 0.0)  #  repeated float prec50_list = 13;
        self.sparsity = opts.pop("sparsity", 0.0)  #  optional float sparsity = 14;
        self.compute_sparsity = opts.pop("computeSparsity", False)  #  optional bool compute_sparsity = 15 [default=false];

        assert len(opts) == 0, opts


class LayerOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.name = opts.pop("name")
        self.file_pattern = opts.pop("filePattern", "")
        self.dimensions = opts.pop("dimensions")
        self.numlabels = opts.pop("numlabels", 1)
        self.param = [ParameterOpts(_) for _ in opts.pop("param", [])]  # repeated deepnet.Parameter param = 5;
        self.is_input = opts.pop("isInput", False)  # optional bool is_input = 7 [default=false];
        self.is_output = opts.pop("isOutput", False)  # optional bool is_output = 8 [default=false];

        self.loss_function = opts.pop("lossFunction",
                                      "SQUARED_LOSS")  # optional LossFunction loss_function = 9 [default=SQUARED_LOSS];
        self.hyperparams = HyperparamsOpts(
            opts.pop("hyperparams", {}))  # optional deepnet.Hyperparams hyperparams = 10;

        self.data_field = DataFieldOpts(opts.pop("dataField", {}))  # optional DataField data_field = 11;
        self.performance_stats = MetricsOpts(
            opts.pop("performanceStats", {}))  # optional deepnet.Metrics performance_stats = 12;
        self.shape = opts.pop("shape", [])  # repeated int32 shape = 13;
        self.is_initialized = opts.pop("isInitialized", False)  # optional bool is_initialized = 14 [default=false];
        self.prefix = opts.pop("prefix", "")  # optional string prefix = 15;
        self.replicate_bias = opts.pop("replicateBias", True)  # optional bool replicate_bias = 16 [default=true];

        # Tying bias parameter.
        self.tied = opts.pop("tied", False)  # optional bool tied = 17 [default=false];
        self.tied_to = opts.pop("tiedTo", "")  # optional string tied_to = 18;

        self.loss_weight = opts.pop("lossWeight", 1.0)  # optional float loss_weight = 19 [default=1.0];

        assert len(opts) == 0


class EdgeOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.node1 = opts.pop("node1")  #  required string node1 = 1;
        self.node2 = opts.pop("node2")  #  required string node2 = 2;
        self.directed = opts.pop("directed", True)  #  optional bool directed = 3 [default=true];
        self.param = [ParameterOpts(_) for _ in opts.pop("param", [])]  #  repeated Parameter param = 4;
        self.hyperparams = HyperparamsOpts(opts.pop("hyperparams", {}))  #  optional Hyperparams hyperparams = 5;
        self.receptive_field_width = opts.pop("receptiveFieldWidth", 1)  #  optional int32 receptive_field_width = 6 [default=1];
        self.display_rows = opts.pop("displayRows", 1)  #  optional int32 display_rows = 7 [default=1];
        self.display_cols = opts.pop("displayCols", 1)  #  optional int32 display_cols = 8 [default=1];
        self.up_factor = opts.pop("upFactor", 1)  #  optional float up_factor = 9 [default=1];
        self.down_factor = opts.pop("downFactor", 1)  #  optional float down_factor = 10 [default=1];
        self.prefix = opts.pop("prefix", "")  #  optional string prefix = 11;
        self.tied = opts.pop("tied", False)  #  optional bool tied = 12[default=false];
        self.tied_to_node1 = opts.pop("tied_to_node1", "")  #  optional string tied_to_node1 = 13;
        self.tied_to_node2 = opts.pop("tied_to_node2", "")  #  optional string tied_to_node2 = 14;
        self.tied_transpose = opts.pop("tied_transpose", False)  #  optional bool tied_transpose = 15[default=false];
        self.block_gradient = opts.pop("block_gradient", False)  #  optional bool block_gradient = 16 [default=false];

        assert len(opts) == 0


class ModelOpts:
    def __init__(self, opts):
        assert isinstance(opts, dict)

        self.name = opts.pop("name")  #  required string name = 1;
        self.model_type = opts.pop("modelType")  #  required ModelType model_type = 2;
        self.layer = [LayerOpts(_) for _ in opts.pop("layer", [])]  #  repeated Layer layer = 3;

        print self.layer
        print self

        self.edge =[EdgeOpts(_) for _ in opts.pop("edge", [])]  #  repeated Edge edge = 4;
        self.hyperparams = HyperparamsOpts(opts.pop("hyperparams", {}))  #  optional Hyperparams hyperparams = 5;
        self.train_stats = [MetricsOpts(_) for _ in opts.pop("trainStats", [])]  #  repeated Metrics train_stats = 6;
        self.validation_stats = [MetricsOpts(_) for _ in opts.pop("validationStats", [])]  #  repeated Metrics validation_stats = 7;
        self.test_stats = [MetricsOpts(_) for _ in opts.pop("testStats", [])]  #  repeated Metrics test_stats = 8;
        self.seed = opts.pop("seed", 0)  #  optional int32 seed = 9 [default=0];

        # For DBMs.
        self.positive_phase_order = opts.pop("positivePhaseOrder", [])  #  repeated string positive_phase_order = 10;
        self.negative_phase_order = opts.pop("negativePhaseOrder", [])  #  repeated string negative_phase_order = 11;
        self.initializer_net = opts.pop("initializerNet", "")  #  optional string initializer_net = 12;
        self.prefix = opts.pop("prefix", "")  #  optional string prefix = 13;

        # Eary Stopping results.
        self.best_valid_stat = MetricsOpts(opts.pop("bestValidStat", {}))  #  optional Metrics best_valid_stat = 14;  // best validation stat
        self.train_stat_es = MetricsOpts(opts.pop("trainStatEs", {}))  #  optional Metrics train_stat_es = 15;  // train stat at best validation
        self.test_stat_es = MetricsOpts(opts.pop("testStatEs", {}))  #  optional Metrics test_stat_es = 16;  // test stat at best validation

        assert len(opts) == 0, opts


def load_missing(dict_main, dict_secondary):
    for k in dict_secondary.keys():
        if k not in dict_main:
            dict_main[k] = dict_secondary[k]


def load_model(net):
    for layer in net["layer"]:
        # layer.hyperparams.MergeFrom(LoadMissing(layer.hyperparams,
        #                                         self.net.hyperparams))

        if "hyperparams" not in layer:
            layer["hyperparams"] = {k: v for (k, v) in net["hyperparams"].items()}

        load_missing(layer["hyperparams"], net["hyperparams"])

    for edge in net["edge"]:
        if "hyperparams" not in edge:
            edge["hyperparams"] = {k: v for (k, v) in net["hyperparams"].items()}

        load_missing(edge["hyperparams"], net["hyperparams"])

    return ModelOpts(net)
