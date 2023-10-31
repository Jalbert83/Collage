from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import torch
import cv2
import glob
import os
#from basicsr.utils import imwrite

import face_recognition

IMPATH = r'.\Aitor'
CROPATH = r'.\AitorCropRecog'


upscale = 2
face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

face_helper.clean_all()

img_list = sorted(glob.glob(os.path.join(IMPATH, '*')))
for img_path in img_list:
    # read image
    img_name = os.path.basename(img_path)
    print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    image = face_recognition.load_image_file(img_path)
    facelocations = face_recognition.face_locations(image)
    for idx, (faceloc) in enumerate(facelocations):
        cropped_face = input_img[faceloc[0]:faceloc[2],faceloc[3]:faceloc[1],:]
        save_crop_path = os.path.join(CROPATH, f'{basename}_{idx:02d}.png')
        cv2.imwrite(save_crop_path, cropped_face)  
    '''
    face_helper.read_image(input_img)
    # get face landmarks for each face
    face_helper.get_face_landmarks_5(only_center_face=False)
    # align and warp each face
    face_helper.align_warp_face()
    for idx, (cropped_face) in enumerate(face_helper.cropped_faces):
        save_crop_path = os.path.join(CROPATH, f'{basename}_{idx:02d}.png')
        cv2.imwrite(save_crop_path, cropped_face)
    '''
