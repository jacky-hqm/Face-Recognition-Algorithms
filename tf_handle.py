# -*- coding:UTF-8 -*-
import os
import sys

# 设置工程路径
curr_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.join(curr_path, "../../")
sys.path.append(project_path)
import numpy as np
import tensorflow as tf



# db 模型加载预测基类
class base_handel():
    def __init__(self, args):
        self.args = args
        if args.session is None:
            self.sess = tf.Session(config=args.sess_cofig)
        else:
            self.sess = args.session
        self.feed_dict = []
        self.predict_list = []

    def _do_load(self):
        # model_file 为模型文件
        # 加载模型
        with tf.gfile.GFile(self.args.model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")
            graph = tf.get_default_graph()
            for index, feed_name in enumerate(self.args.feed_name_list):
                self.feed_dict.append(graph.get_operation_by_name(feed_name).outputs[0])
            for predict_name in self.args.out_name_list:
                self.predict_list.append(graph.get_operation_by_name(predict_name).outputs[0])
        return self

    def _do_predict(self, feed_data):
        feed_data = np.array(feed_data)
        if len(feed_data.shape) < 5:
            feed_data = np.expand_dims(feed_data, axis=0)
        feed_dict = {ii: dd for ii, dd in zip(self.feed_dict, feed_data)}
        predict_res = self.sess.run(self.predict_list, feed_dict=feed_dict)
        return predict_res

    def _do_validate_dir(self, dir, dir_label):
        import cv2
        import vic_base
        usrful_ext = vic_base.img_format()
        image_names = os.listdir(dir)
        result_list = []
        for img_name in image_names:
            file_extension = os.path.splitext(img_name)[1]
            if file_extension.lower() in usrful_ext:
                image = cv2.imdecode(np.fromfile(os.path.join(dir, img_name), dtype=np.uint8),
                                     cv2.IMREAD_UNCHANGED)
                if image is None:
                    print('img read failed : %s', os.path.join(dir, img_name))
                    continue
                result = self._do_predict(self, [image])
                result_list.append(np.argmax(result[0]))
        print(result_list)