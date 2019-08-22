# -*- coding:UTF-8 -*-
# -*- coding:UTF-8 -*-
import os
import sys

# 设置工程路径
curr_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.join(curr_path, "../../")
sys.path.append(project_path)

# 确定训练使用的网络
net_name = 'shufflenet'
# 加载网络方法
import vic_base
import cv2
import numpy
import argparse
import tensorflow as tf
import logger
#可以把自己模块封装成类导入
from importlib import import_module

net = import_module(net_name)
logging = logger._getlogger('tf_age_gender_train')

#保存图片数据为tfrecords格式
def quality_img_to_tfrecord(img_root_path, save_prefix, useful_dirs, resize=None, shuffle=True, dir_class=False):
    """
    transform img to data tfrecords.
    Args:
        img_root_path: img_path
        save_prefix: .tfrecords save
        useful_dirs:指定有用目录 ['frontal', 'bigangle_left', 'bigangle_right', 'notface']
        resize:size for resizing img
        shuffle:shuffle the label and img data
        specify_num:specify class num
        save_name: name of .tfrecords
    """
    usrful_ext = vic_base.img_format()#['.jpg', 'jpeg', '.png', '.bmp']
    dir_nums = os.listdir(img_root_path)
    examples_nums = 0
    class_nums = 0
    image_list = []
    label_list = []
    for index, name in enumerate(dir_nums):
        if name in useful_dirs:#['frontal', 'bigangle_left', 'bigangle_right', 'notface']
            class_path = os.path.join(img_root_path, name)#种类路径
            single_label = int(useful_dirs.index(name))#index() 函数用于从列表中找出某个值第一个匹配项的索引位置
            #print(name)#['frontal', 'bigangle_left', 'bigangle_right', 'notface']
            if os.path.isdir(class_path):#读取class_path下的所有图片
                class_nums += 1#种类数加一
                for img_name in os.listdir(class_path):#获取图片名
                    file_extension = os.path.splitext(img_name)[1]#分离文件名与扩展名，获取扩展名就是.jpg
                    if file_extension.lower() in usrful_ext:# lower() 方法转换字符串中所有大写字符为小写。
                        image_list.append(os.path.join(name, img_name))
                        if dir_class:
                            label_list.append(int(name))
                        else:
                            label_list.append(int(single_label))
                        examples_nums += 1

    logger.vic_logger.info('共 %d 类，样本数量共 %d', class_nums, examples_nums)
    # print(image_list)#bigangle_left\\0-FaceId-0.jpg',.....
    # print(label_list)#[1,1.....

    if shuffle:
        temp = numpy.array([image_list, label_list])
        temp = temp.transpose()#把矩阵转成2X14412的矩阵
        numpy.random.shuffle(temp)#把列表打乱
        # 从打乱的temp中再取出list（img和lab）
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
    #设置保存路径和文件名
    writer_path = save_prefix + '_C' + str(class_nums) + '_N' + str(
        examples_nums) + '_' + str(resize) + '_img_data.tfrecords'
    #用来向模型中写入数据
    writer = tf.python_io.TFRecordWriter(writer_path)#该模块是tensorflow用来处理tfrecords文件的接口，将记录写入TFRecords文件的类。
    for i in range(len(image_list)):
        # logger.vic_logger.debug('num = %d，%s, %s', i, image_list[i], label_list[i])
        img = cv2.imread(os.path.join(img_root_path, image_list[i]),cv2.IMREAD_GRAYSCALE)#读取每一张图片
        if img is None:
            print('读取失败')
            continue
        if resize is not None:#如果设置了图片分辨率大小
            img = cv2.resize(img, resize)
        # logger.vic_logger.info('img 的 type %s', img.dtype)
        img_raw = img.tobytes()  # 将图片转为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_list[i])])),
            "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())#序列化为字符串
        if i % 1000 == 0:
            logger.vic_logger.info('num = %d', i)
    writer.close()

#Model 继承net.NetAlgo
class Model(net.NetAlgo):
    def train_private_args(self, parser):
        # 设定各个训练独有的参数
        private_train = parser.add_argument_group('Private', 'Private train parser')
        private_train.add_argument('--l2_strength', type=float, default=4e-5,
                                   help='the l2_strength of shufflenet')
        private_train.add_argument('--bias', type=float, default=0.0,
                                   help='the bias of shufflenet')
        private_train.add_argument('--bottleneck-layer-flag', type=bool, default=False,
                                   help='bottleneck_layer_flag of shufflenet')
        private_train.add_argument('--bottleneck-layer-size', type=int, help='num of net size before full_connected')

        private_train.add_argument('--batchnorm-enabled', type=bool, default=True, help='batchnorm enable flag ')

        private_train.add_argument('--num-groups', type=int, help='the group num of shufflenet')
        private_train.add_argument('--stage-repeats', type=int, help='stage_repeats')

        # private_train.add_argument('--bn-decay', type=float, default=0.995, help='the bn_decay of mobilenet_v1')
        # private_train.add_argument('--bn-epsilon', type=float, default=0.001,
        #                            help='the bn-epsilon of mobilenet_v1')
        # private_train.add_argument('--weight-decay', type=float, default=0.0,
        #                            help='the weight-decay of mobilenet_v1')

        private_train.add_argument('--data-train-tfrecords', type=str, help='train data tfrecords path')
        private_train.add_argument('--data-train-tfrecords-shape', help='train data img shape')
        private_train.add_argument('--data-train-tfrecord', type=str, help='train data tfrecord path')
        private_train.add_argument('--data-val-img', type=str, help='val data img path')
        private_train.add_argument('--data-val-tfrecord', type=str, help='val data tfrecord path')
        private_train.add_argument('--log-path', type=str, help='log path')
        return private_train

    def _do_parse_params(self, parameter):
        '''
        将参数转换成内部变量
        '''
        self.network = parameter.network
        self.mode_type = parameter.mode_type
        if self.mode_type == 'train':
            self.phase_train = True
            logging.info('Train Mode')
        else:
            self.phase_train = False
            logging.info('Test Mode')
        self.num_classes = parameter.num_classes
        self.img_width = parameter.img_width
        self.img_height = parameter.img_height
        self.img_channel = parameter.img_channel
        self.img_shape = (self.img_height, self.img_width, self.img_channel)
        self.data_train_tfrecords = parameter.data_train_tfrecords
        self.data_train_tfrecords_shape = parameter.data_train_tfrecords_shape
        self.num_epochs = parameter.num_epochs
        self.lr = parameter.lr
        self.batch_size = parameter.batch_size
        self.bottleneck_layer_flag = parameter.bottleneck_layer_flag
        self.bottleneck_layer_size = parameter.bottleneck_layer_size
        self.l2_strength = parameter.l2_strength
        self.bias = parameter.bias
        self.batchnorm_enabled = parameter.batchnorm_enabled
        self.num_groups = parameter.num_groups
        self.stage_repeats = parameter.stage_repeats
        # self.dropout_keep_prob = parameter.dropout_keep_prob
        # self.bn_decay = parameter.bn_decay
        # self.bn_epsilon = parameter.bn_epsilon
        # self.weight_decay = parameter.weight_decay
        self.fix_params = parameter.fix_params
        self.log_path = parameter.log_path
        self.gpus = parameter.gpus
        self.fine_tune = parameter.fine_tune
        self.disp_batches = parameter.disp_batches
        self.snapshot_iters = parameter.snapshot_iters
        self.model_prefix = parameter.model_prefix


def train_quality_judge():
    # 设定参数
    parser = argparse.ArgumentParser(description="train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common_model = Model(parser)

    # 设定确切的值
    parser.set_defaults(
        # public
        mode_type='train',  # 指定运行模式 train 、predict 、feature
        network=net_name,
        num_classes=4,
        img_width=112,  # 图片宽度
        img_height=112,  # 图片高度
        img_channel=3,  # 图片通道数
        num_epochs=10,  # 训练epoch 个数，运行次数
        lr=0.001,  # 设置学习率
        bottleneck_layer_flag=False,
        bottleneck_layer_size=128,
        l2_strength=4e-5,
        bias=0.0,
        num_groups=1,
        stage_repeats=[3, 7, 3],
        # bn_decay=0.9,
        batch_size=100,  # 设定batch size每个循环中你所输入的数据块的大小。
        # dropout_keep_prob=1.0,
        fix_params=None,
        # data_train_tfrecords=r'/lianlian/data/ms-celeb-1m-crop/ms_tfrecords/ms_C64260_N4497730_img_data.tfrecords',
        data_train_tfrecords=r'D:\svm\tfrecords\gray\gray_C4_N17998_(171, 171)_img_data.tfrecords',
        data_train_tfrecords_shape=(171, 171,3),
        log_path=r'./models/shufflenet3',
        gpus='0',
        disp_batches=10,  # 显示logging信息，每个10次显示运行日志
        snapshot_iters=100,  # 保存临时模型，每隔100次迭代保存一次模型
        model_prefix='tf_shufflenet'
    )
    # 生成参数
    args = parser.parse_args()
    with tf.Graph().as_default():
        common_model.build_graph(args)
        common_model.build_train_op()
        common_model.train()

def pack_pb_model():
    from tf_pack import pack_db_model
    model_file = r'.\models\shufflenet3\tf_shufflenet'
    model_iter_file=r'.\models\shufflenet3\tf_shufflenet_iter_600'
    pack_handle = pack_db_model()
    pack_handle.transform_to_db(model_iter_file,
                                [net_name + "/logits/BiasAdd"])
if __name__ == '__main__':
    #预测结果：    0          1                  2                3
    train_label = ['frontal', 'bigangle_left', 'bigangle_right', 'notface']

    #data 准备
    img_root_path = r'\\LT-CHENXIN\yunshike\face_samples\train'
    save_prefix = r'D:\svm\tfrecords\gray'
    #quality_img_to_tfrecord(img_root_path, save_prefix,train_label, resize=(171, 171))

    #train_quality_judge()
    pack_pb_model()
