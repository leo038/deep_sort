import torch
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tools.generate_detections import create_box_encoder


def object_detect(img, model):
    result = model(img)
    pred = result.pred[0]  # list of tensors pred[0] = (xyxy, conf, cls)
    ## select only person
    pred_person = pred[pred[:, 5] == 0]


    ## convert the result to MOTchallenge format
    pred_person = pred_person.cpu().numpy()
    object_num = len(pred_person)
    mot_det = np.ones(shape=(object_num, 10)) * -1
    mot_det[:, 2:7] = pred_person[:, :5]
    mot_det[:, 4:6] = pred_person[:, 2:4] - pred_person[:, :2]
    # print(pred_person)
    # print(mot_det)

    return mot_det


def get_reid_features(img_file, detect_result):
    encoder = create_box_encoder(model_filename="mars.pb", batch_size=64)

    bgr_image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    features = encoder(bgr_image, detect_result[:, 2:6])
    print(f"reid features: {features.shape}")
    return features


def get_det_and_features(img_file, model):
    img = cv2.imread(img_file)
    img = img[:, :, ::-1]  ##BGR to RGB
    detect_result = object_detect(img, model=model)
    features = get_reid_features(img_file, detect_result)

    merge_result =  np.c_[detect_result, features]  #[np.r_[(det, feature)] for det, feature in zip(detect_result, features)]
    print(f"merge result: {merge_result.shape}")
    return merge_result


if __name__ == "__main__":
    img_dir = "/data/joyiot/liyong/datasets/MOT16/test/MOT16-06/img1"
    file_list = os.listdir(img_dir)
    model = torch.hub.load('/data/joyiot/.cache/torch/hub/ultralytics_yolov5_master/', 'yolov5x', source='local')


    detections_out = []
    for index, file_name in enumerate(file_list):
        print(f"Process {index}/{len(file_list)} image.")
        img_file = os.path.join(img_dir, file_name)
        out = get_det_and_features(img_file, model)
        detections_out += [out]
        if index>50:
            break
    np.save('./offline_det.npy',  np.asarray(detections_out), allow_pickle=True)
# #  1,-1,1344,386,30.903,70.127,85.914,-1,-1,-1
#     res = np.load("./offline_det.npy", allow_pickle=True)
#     print(res[0])