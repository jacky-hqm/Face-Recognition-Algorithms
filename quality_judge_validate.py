# -*- coding:UTF-8 -*-
import numpy as np
import os
import cv2
import tensorflow as tf
from  tf_handle import base_handel
import time
from easydict import EasyDict as edict
import vic_base
import shutil
from sklearn.externals import joblib
# 查看模型的准确性
from sklearn.metrics import confusion_matrix
# classification_report看分类报告的结果
from sklearn.metrics import classification_report


class ZQQualityJudge(base_handel):
    def _do_predict(self, feed_data):
        feed_data = np.expand_dims(feed_data, axis=0)
        feed_data = [feed_data, False]
        feed_dict = {ii: dd for ii, dd in zip(self.feed_dict, feed_data)}
        result = self.sess.run(self.predict_list, feed_dict=feed_dict)
        result = np.argmax(result[0])
        return result



if __name__ == '__main__':
    # 配置参数
    ga_args = edict()
    ga_args.sess_cofig = tf.ConfigProto()
    ga_args.session = tf.Session(config=ga_args.sess_cofig)
    ga_args.model_file = './models/shufflenet2/tf_shufflenet.pb'
    ga_args.feed_name_list = ['shufflenet/inputs_placeholder', 'shufflenet/phase_train_placeholder']
    ga_args.out_name_list = ['shufflenet/logits/BiasAdd']

    # 设置文件路径
    # train_dir = r'\\LT-CHENXIN\yunshike\face_samples\train'
    # train_dir = r'D:\svm\test'
    # test_dir = r'\\LT-CHENXIN\yunshike\face_samples\test'
    # # 获取图片和标签列表
    # image_list, label_list = get_files(test_dir)
    # print(len(image_list))
    # print(len(label_list))
    # 加载模型
    with tf.device('/gpu:0'):
        quality_judge_detector = ZQQualityJudge(ga_args)._do_load()

        # path1='20180731170432Rf7j2upw_d9y34f.jpg'
        # path2='00000001-20-10-20-0003.jpg'
        # path3='00000000-17-01-17-0002.jpg'
        # path4='00000016-18-50-58-0004.jpg'
        # path=[path1,path2,path3,path4]
        # for i in path:
        #     img1 = cv2.imread(i)
        #     start_time = time.time()
        #     face_img = cv2.resize(img1, (112, 112))
        #     result=quality_judge_detector._do_predict(face_img)
        #     print(result)













        # save_root_path = r'E:\workspace\imagedata\yunshike\result'
        # label_list = ['0', '1', '2', '3']
        # for label in label_list:
        #     if not os.path.exists(os.path.join(save_root_path, label)):
        #         os.makedirs(os.path.join(save_root_path, label))
        #
        # # 文件夹分类
        # if True:
        #     img_root_path = r'F:\mkh\imgs2018_07_31'
        #     save_root_path = r'E:\workspace\imagedata\yunshike\to_selected\mkh\face_img'
        #     save_ysq_path = r'E:\workspace\imagedata\yunshike\to_selected\mkh\face_img_2'
        #     mtcnn_face_num = 0
        #     ysq_face_num = 0
        #
        #     for root, dirs, files in os.walk(img_root_path, topdown=False):
        #         for file in files:
        #             if os.path.splitext(file)[1].lower() in vic_base.img_format():
        #                 img = cv2.imread(os.path.join(root, file))
        #                 if img is None:
        #                     print('read img failed : %s' % (os.path.join(root, file)))
        #                     continue
        #                 else:
        #                     start_time = time.time()
        #                     face_imgs, bounding_boxes, points = mtcnn_handel.detect_align(img, minsize=40,
        #                                                                                   choose_center_face=False,
        #                                                                                   return_contain_box=True)
        #                     print('mtcnn spend ', (time.time() - start_time) * 1000)
        #                     I = img.copy()
        #                     if len(face_imgs) == 0:
        #                         print('mtcnn no face')
        #                         # continue
        #                     else:
        #                         for i in range(bounding_boxes.shape[0]):
        #                             box, pts = np.int32(bounding_boxes[i]), np.int32(points[:, i])
        #                             I = cv2.rectangle(I, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)
        #                             for i in range(5):
        #                                 I = cv2.circle(I, (pts[i], pts[i + 5]), 2, (255, 255, 255), thickness=-1)
        #                         # 保存图片
        #                         mtcnn_face_num += 1
        #                         shutil.move(os.path.join(root, file), os.path.join(save_root_path, file))
        #
        #                     # cv2.imshow('mtcnn', cv2.resize(I, (640, 480)))
        #
        #                     start_time = time.time()
        #                     face_boxes, face_datas = ysq_handel.detect_face(img, minsize=40)
        #                     print('ysq spend ', (time.time() - start_time) * 1000)
        #                     if face_boxes is None:
        #                         print('ysq no face')
        #                     else:
        #                         ysq_face_num += 1
        #                         if os.path.exists(os.path.join(root, file)):
        #                             shutil.move(os.path.join(root, file), os.path.join(save_ysq_path, file))
        #
        #                         for index, face_data in enumerate(face_datas):
        #                             start_time = time.time()
        #                             #读图片
        #                             face_img = cv2.resize(face_data, (112, 112))
        #                             result = quality_judge_detector._do_predict(face_img)
                                    # print(result, 'spend ', (time.time() - start_time) * 1000)
                                    # shutil.copy(os.path.join(root, file), os.path.join(save_root_path, str(result)))

                            # cv2.imshow('ysq', cv2.resize(img, (640, 480)))
                            # cv2.waitKey(1)
        # if False:
        #     # 视频检测
        #     cap = cv2.VideoCapture(0)
        #
        #     cap = cv2.VideoCapture('rtsp://admin:zhiqu123@10.20.37.185/cam/realmonitor?channel=1&subtype=0')
        #     show_rlt = True
        #     while True:
        #         # get video frame
        #         ret, img = cap.read()
        #         face_imgs, bounding_boxes, points = mtcnn_handel.detect_align(img, minsize=40, choose_center_face=False,
        #                                                                       return_contain_box=True)
        #         result_list = []
        #         if len(face_imgs) == 0:
        #             pass
        #         else:
        #             for face_img in face_imgs:
        #                 start_time = time.time()
        #                 face_img = cv2.resize(face_img, (112, 112))
        #                 result_list.append(quality_judge_detector._do_predict(face_img))
        #                 print('spend', (time.time() - start_time) * 1000)
        #         if show_rlt:
        #             for i in range(bounding_boxes.shape[0]):
        #                 box, pts = np.int32(bounding_boxes[i]), np.int32(points[:, i])
        #                 img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
        #                 for j in range(5):
        #                     img = cv2.circle(img, (pts[j], pts[j + 5]), 2, (255, 255, 255), thickness=-1)
        #                 label = "{}".format(str(result_list[i]))
        #                 cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #             img = cv2.resize(img, (640, 480))
        #             cv2.imshow('Capture', img)
        #             cv2.waitKey(10)
    #
    # # # 加载图片预测
    # # img = cv2.imread('../../data/rec_imgs/1-1.jpg')
    # # img = cv2.resize(img, (112, 112))
    # # result = quality_judge_detector._do_predict(img)
    # # print(result)
    # # # 加载图片预测
    # # img = cv2.imread('../../data/rec_imgs/2-1.jpg')
    # # img = cv2.resize(img, (112, 112))
    # # result = quality_judge_detector._do_predict(img)
    # # print(result)
    # # # 加载图片预测
    # # img = cv2.imread('../../data/rec_imgs/2-3.jpg')
    # # img = cv2.resize(img, (112, 112))
    # # result = quality_judge_detector._do_predict(img)
    # # print(result)
    #
