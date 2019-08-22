# -*- coding:UTF-8 -*-
import os
import tensorflow as tf


class pack_db_model():
    def __init__(self):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()

    def load_model(self, model_file):
        saver = tf.train.import_meta_graph(model_file + '.meta', clear_devices=True)
        saver.restore(self.sess, model_file)

        variable_names = [v.name for v in tf.all_variables()]
        for variable_name in variable_names:
            print(variable_name)

        # images_placeholder = self.graph.get_tensor_by_name("tower_1/image_batch:0")
        # embeddings = self.graph.get_tensor_by_name("tower_1/embeddings:0")
        # phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")

    def transform_to_db(self, model_file, last_layer_names, save_model_name=None):
        self.load_model(model_file)
        for index, last_layer_name in enumerate(last_layer_names):
            rindex = last_layer_name.rfind(':')
            if rindex >= 0:
                last_layer_name = last_layer_name[:rindex]
            last_layer_names[index] = last_layer_name
        c = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, last_layer_names)
        model_dir, model_name = os.path.split(model_file)
        if save_model_name is None:
            save_model_name = model_name + '.pb'
        tf.train.write_graph(c, model_dir, save_model_name, as_text=False)


if __name__ == '__main__':
    # 指定模型所在路径
    # 指定输出层名
    model_file = r'../../models/insightface/InsightFace_iter_best_710000.ckpt'
    pack_handle = pack_db_model()
    pack_handle.transform_to_db(model_file, ['resnet_v1_50/E_BN2/Identity'])
