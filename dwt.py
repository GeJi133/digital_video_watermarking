import cv2
import numpy as np
import os
import pywt
import math
import random
from attack import VideoAttack


def get_moving_block(frame_YUV_list, block_size, alpha=0.4, belta=0.6):
    print("选择嵌入块...")
    frame_num = len(frame_YUV_list)
    frame_h = frame_YUV_list[0].shape[0]
    frame_w = frame_YUV_list[0].shape[1]
    block_num = int(frame_num / block_size)
    score_list = []
    for b in range(block_num):
        vk_block = np.zeros(shape=(frame_h, frame_w))
        mean_block = np.zeros(shape=(frame_h, frame_w, block_size))
        mk_list = [np.mean(frame_YUV_list[i]) for i in range(frame_num)]
        print("处理第" + str(b + 1) + "快/" + str(block_num))
        for i in range(frame_h):
            for j in range(frame_w):
                vk_list = [
                    abs(
                        int(frame_YUV_list[b * block_size + kk + 1][i, j]) -
                        int(frame_YUV_list[b * block_size + kk][i, j])
                    )
                    for kk in range(block_size - 1)
                ]
                vk_block[i, j] = max(vk_list)
                for kk in range(block_size):
                    mean_block[i, j, kk] = (1 / mk_list[kk]) ** 0.6 * (
                            abs(frame_YUV_list[kk][i, j] - mk_list[kk]) / mk_list[kk])
        mk_ave = np.mean(mean_block)
        vk_ave = np.mean(vk_block)
        score_list.append(belta * vk_ave + alpha * (np.mean(mk_list) + mk_ave))
    max_value = max(score_list)
    blcok_index = score_list.index(max_value)
    # frame_block=frame_YUV_list[blcok_index*block_size:(blcok_index+1)*block_size]
    return (blcok_index * block_size, (blcok_index + 1) * block_size)


def setwatermark(water_img, frame_u_list, frame_num, q=200):
    print('水印嵌入...')
    water_h = water_img.shape[0]
    water_w = water_img.shape[1]
    cD3_block = np.zeros(shape=(water_h, water_w, frame_num))
    dwt_list = []

    for i in range(frame_num):
        c = pywt.wavedec2(frame_u_list[i], 'db2', level=3)
        [cl, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = c
        cD3_water_part = cD3[0:water_h, 0:water_w]
        cD3_block[..., i] = cD3_water_part
        dwt_list.append(c)
    db1 = pywt.Wavelet('db1')
    for i in range(water_h):
        for j in range(water_w):
            time_line = cD3_block[i, j]
            cA3, cD3, cD2, cD1 = pywt.wavedec(time_line, db1, level=3)
            ave = np.mean(cA3)
            z = math.floor(ave / q + 0.5)
            if z % 2 != water_img[i][j]:
                if (0 == math.floor(ave / q)):
                    for k in range(len(cA3)):
                        cA3[k] = (cA3[k] + q)
                else:
                    for k in range(len(cA3)):
                        cA3[k] = (cA3[k] - q)
            coeffs = [cA3, cD3, cD2, cD1]
            out = pywt.waverec(coeffs, db1, mode='constant')

            for kk in range(frame_num):
                dwt_list[kk][1][2][i][j] = out[kk]
    # 重构
    new_frame_YUV_list = []
    for i in range(frame_num):
        newImg = pywt.waverec2(dwt_list[i], 'db2')
        newImg = np.array(newImg, np.uint8)
        new_frame_YUV_list.append(newImg)
    return new_frame_YUV_list


def getwatermark(frame_u_list, waterImgShape, q=200):
    print('水印提取...')
    water_h = waterImgShape[0]
    water_w = waterImgShape[1]
    frame_num = len(frame_u_list)
    cD3_block = np.zeros(shape=(water_h, water_w, frame_num))
    dwt_list = []
    for i in range(frame_num):
        c = pywt.wavedec2(frame_u_list[i], 'db2', level=3)
        [cl, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = c
        cD3_water_part = cD3[0:water_h, 0:water_w]
        cD3_block[..., i] = cD3_water_part
        dwt_list.append(c)
    db1 = pywt.Wavelet('db1')
    waterImg = np.zeros(shape=(water_h, water_w))
    for i in range(water_h):
        for j in range(water_w):
            time_line = cD3_block[i, j]
            cA3, cD3, cD2, cD1 = pywt.wavedec(time_line, db1, mode='constant', level=3)
            ave = np.mean(cA3)
            z = math.floor(ave / q + 0.5)
            waterImg[i][j] = z % 2
            # print("len(cA3_out)",len(out),out)
    return waterImg


def get_wrong_rate(set_watermark, get_watermark):
    watermark_h = set_watermark.shape[0]
    watermark_w = set_watermark.shape[1]
    right_num = 0
    for i in range(watermark_h):
        for j in range(watermark_w):
            if (set_watermark[i, j] == get_watermark[i, j]):
                right_num += 1
    return right_num / (watermark_w * watermark_h)


def set_syn(frame_list, start_num, end_num, key):
    for i in range(start_num, end_num):
        for kk in range(len(key)):
            if frame_list[i][0][kk] % 2 != key[kk]:
                frame_list[i][0][kk] += 1
    return frame_list


def check(M_a, M_b, T):
    r = 0
    for i in range(len(M_a)):
        r = r + M_a[i] * M_b[i]
    if (r > T):
        return True
    else:
        return False


def get_syn(frame_list, m):
    water_list = []
    m = [-1 if x == 0 else 1 for x in m]
    for i in range(len(frame_list)):
        a = [-1 if x % 2 == 0 else 1 for x in frame_list[i][0][:len(m)]]
        if check(a, m, 14):
            water_list.append(frame_list[i])
    return water_list


if __name__ == '__main__':
    # 读取视频
    capture = cv2.VideoCapture("./dataset/video/poniu1036.mp4")
    assert capture.isOpened(), "视频读入失败，请检查文件路径"
    v_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fp_nums = int(capture.get(7))
    video_data = np.zeros(shape=(v_height, v_width, fp_nums))
    grey_frame_YUV_list = []  # 图像灰度矩阵，用于选择嵌入块
    frame_YUV_list = []  # 帧YUV格式列表
    frame_u_list = []  # 帧YUV中U值列表
    for i in range(fp_nums):
        ret, img = capture.read()  # img 就是一帧图片
        if not ret: break
        background = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yuv_background = cv2.cvtColor(background, cv2.COLOR_RGB2YUV)  # 将RBG格式的背景转为YUV格式，Y为灰度层，U\V为色彩层，此处选择U层进行嵌入
        frame_YUV_list.append(yuv_background)
        grey_frame_YUV_list.append(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
        frame_u_list.append(yuv_background[..., 1])
    # 读取水印信息
    waterImg = cv2.imread('./dataset/watermark/xiaohui130_130.png')
    water_img = cv2.cvtColor(waterImg, cv2.COLOR_RGB2GRAY)
    water_img = np.where(water_img < np.mean(water_img), 0, 1)

    # 保存嵌入前水印图像
    cv2.imwrite("./dataset/getwater/water_img.png", water_img * 255)

    # (start_frame,end_frame) = get_moving_block(grey_frame_YUV_list, frame_num)
    (start_frame, end_frame) = (60, 80)
    frame_num = end_frame - start_frame  # 嵌入水印帧数
    # key =[1,0,0,1,1,0,1,0,1,1,1,1,0,0,0]
    # frame_u_list=set_syn(frame_u_list,start_frame,end_frame,key)

    new_frame_YUV_list = setwatermark(water_img, frame_u_list[start_frame:end_frame], frame_num)
    frame_RGB_list = np.copy(frame_YUV_list)
    # 还原视频
    for i in range(len(frame_YUV_list)):
        if i in range(start_frame, end_frame):
            # frame_YUV_list[i]=new_frame_YUV_list[i-start_frame]
            yuv_background = frame_YUV_list[i]
            synthesis = new_frame_YUV_list[i - start_frame]
            yuv_background[..., 1] = synthesis
        else:
            yuv_background = frame_YUV_list[i]
        rbg_synthesis = cv2.cvtColor(yuv_background, cv2.COLOR_YUV2RGB)
        rbg_synthesis = cv2.cvtColor(rbg_synthesis, cv2.COLOR_RGB2BGR)
        frame_RGB_list[i] = rbg_synthesis
    # 将加入水印视频写入文件
    v_writer = cv2.VideoWriter("./dataset/video/poniu1036_attack.mp4", fourcc, fps, (v_width, v_height))
    for kk in range(len(frame_RGB_list)):
        v_writer.write(frame_RGB_list[kk])
    # 加噪
    video_attack = VideoAttack()
    frame_RGB_list[start_frame:end_frame] = video_attack.Add_Gaussian(frame_RGB_list[start_frame:end_frame], 0, 100)
    # 从视频中提取水印
    frame_list_water = frame_RGB_list
    frame_u_list = []
    for i in range(len(frame_RGB_list)):
        frame_rgb = cv2.cvtColor(frame_RGB_list[i], cv2.COLOR_BGR2RGB)
        yuv_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YUV)  # 将RBG格式的背景转为YUV格式，Y为灰度层，U\V为色彩层，此处选择U层进行嵌入
        frame_u_list.append(yuv_frame[..., 1])
    # insert_water_frame = get_syn(frame_u_list, key)
    insert_water_frame = frame_u_list[start_frame:end_frame]
    # 抽帧攻击
    # delete_rate=0.3
    # frame_num=len(insert_water_frame)
    # for kk in range(frame_num):
    #     if random.random()<delete_rate:
    #         insert_water_frame.pop(frame_num-kk-1)
    waterImg = getwatermark(insert_water_frame, [waterImg.shape[0], waterImg.shape[1]])
    print("误码率", get_wrong_rate(waterImg, water_img))
