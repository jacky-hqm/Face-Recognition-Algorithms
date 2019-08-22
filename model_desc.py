"""
    tensorflow 网络基类

"""
import os
import glob



# tensorflow 网络最基本的类
class ModelDescBase(object):
    """
    Base class for a models description.
    """

    # 指定输入占位符
    def _do_inputs(self):
        """
        __Create__ and returns a list of placeholders.
        A subclass is expected to implement this method.

        The placeholders __have to__ be created inside this method.
        Don't return placeholders created in other methods.
        Also, you should not call this method by yourself.

        Returns:
            a list of `tf.placeholder`, to be converted to :class:`InputDesc`.
        """
        raise NotImplementedError()

    # 构建图
    def build_graph(self, args):
        """
        Build the whole symbolic graph.
        This is supposed to be part of the "tower function" when used with :class:`TowerTrainer`.
        By default it will call :meth:`_build_graph` with a list of input tensors.

        A subclass is expected to overwrite this method or the :meth:`_build_graph` method.

        Args:
            args ([tf.Tensor]): tensors that matches the list of inputs defined by ``inputs()``.

        Returns:
            In general it returns nothing, but a subclass (e.g.
            :class:`ModelDesc`) may require it to return necessary information
            (e.g. cost) to build the trainer.
        """
        raise NotImplementedError()

    def build_train_op(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


# 单独 loss 和 optimizer 的类
class ModelDesc(ModelDescBase):
    """
    A ModelDesc with **single cost** and **single optimizer**.
    It has the following constraints in addition to :class:`ModelDescBase`:

    1. :meth:`build_graph(...)` method should return a cost when called under a training context.
       The cost will be the final cost to be optimized by the optimizer.
       Therefore it should include necessary regularization.

    2. Subclass is expected to implement :meth:`optimizer()` method.

    """

    def _do_cost(self, logits, labels):
        raise NotImplementedError()

    def _do_optimizer(self):
        """
        Returns a `tf.train.Optimizer` instance.
        A subclass is expected to implement this method.
        """
        raise NotImplementedError()

    def _do_evaluate(self, logits, labels):
        raise NotImplementedError()

    # 设定公用的参数
    def train_public_args(self, parser):
        public_train = parser.add_argument_group('Public', 'Public train parser')
        # 模式 train、predict、feature
        public_train.add_argument('--mode-type', type=str, help='the mode type to do')
        # 训练使用的网络
        public_train.add_argument('--network', type=str, help='the neural network to use')
        # 分类的个数
        public_train.add_argument('--num-classes', type=int, help='the number of classes')
        # 图片 width
        public_train.add_argument('--img-width', type=int, help='train img width.')
        # 图片 height
        public_train.add_argument('--img-height', type=int, help='train img height.')
        # 图片 channel
        public_train.add_argument('--img-channel', type=int, help='train img channel.')
        # 训练样本的个数
        public_train.add_argument('--num-examples', type=int, help='the number of training examples')
        # 训练epoch个数
        public_train.add_argument('--num-epochs', type=int, default=100, help='max num of epochs')
        # 学习率 lr
        public_train.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
        # batch_size 预设值
        public_train.add_argument('--batch-size', type=int, default=128, help='the batch size')
        # 是否finetune
        public_train.add_argument('--fine-tune', type=bool, default=False, help='fine tune flag')
        # 是否固定参数
        public_train.add_argument('--fix-params', type=int, default=None, help='fix params flag')
        # 模型前缀（路径加前缀）
        public_train.add_argument('--models-prefix', type=str, help='models prefix')
        # 训练使用的GPU 资源 [0,1,2,3]
        public_train.add_argument('--gpus', type=str,
                                  help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
        # 运行过程中多少个batch_size 打印信息
        public_train.add_argument('--disp-batches', type=int, default=40, help='show progress for every n batches')

        # 运行过程中多少个batch_size 保存模型
        public_train.add_argument('--snapshot-iters', type=int, default=100, help='snapshot iters for every n batches')

        #  # 输入数据的类型
        #  public_train.add_argument('--dtype', type=str, default='float32', help='the type of input data')

        #  # 初始化方法
        #  public_train.add_argument('--initializer', type=str, default='default', help='the initializer type')
        #  # 优化器方法 默认为SGD
        #  public_train.add_argument('--optimizer', type=str, default='sgd', help='the optimizer type')
        #  # SGD momentum 默认值
        #  public_train.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
        #  # SGD momentum 默认值
        #  public_train.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
        #

        #  # 指定加载哪个epoch的模型
        #  public_train.add_argument('--load-epoch', type=int,
        #                            help='load the models on an epoch using the models-load-prefix')
        #
        #  # 指定epoch将参数求log值
        #  public_train.add_argument('--monitor', dest='monitor', type=int, default=0,
        #                            help='log network parameters every N iters if larger than 0')
        #  # 指定feature 模式下需要获取的层的计算结果
        #  public_train.add_argument('--specify-layer', type=str, help='the specify layer of feature mode get feature')
        return public_train

    # 设定私有的参数
    def train_private_args(self, parser):
        raise NotImplementedError()

    def find_previous(self, model_prefix):
        sfiles = model_prefix + '_iter_*.meta'
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        sfiles = [ss.replace('.meta', '') for ss in sfiles]

        for sfile in sfiles:
            print('sfile', sfile)

        nfiles = model_prefix + '_iter_*.index'
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)

        lsf = len(sfiles)
        assert len(nfiles) == lsf
        return lsf, sfiles
