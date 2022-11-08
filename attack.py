import cv2
import numpy as np
class VideoAttack():
    def __init(self):
        pass
    @staticmethod
    def Add_SaltPepper(frame_list):
        attacked_frames=[]
        print("add_saltPepper")
        for image in frame_list:
            s_vs_p = 0.5
            # 设置添加噪声图像像素的数目
            amount = 0.04
            noisy_img = np.copy(image)
            # 添加salt噪声
            num_salt = np.ceil(amount * image.size * s_vs_p)
            # 设置添加噪声的坐标位置
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_img[coords] = 255
            # 添加pepper噪声
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            # 设置添加噪声的坐标位置
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_img[coords] = 0
            attacked_frames.append(noisy_img)
        return attacked_frames
    @staticmethod
    def Add_Gaussian(frame_list,mean,sigma):
        attacked_frames=[]
        print("加噪")
        for image in frame_list:
            # print("加噪1223")
            img_height=image.shape[0]
            img_width=image.shape[1]
            img_channels=image.shape[2]
            # mean = 0
            # 设置高斯分布的标准差
            # 根据均值和标准差生成符合高斯分布的噪声
            gauss = np.random.normal(mean, sigma, (img_height, img_width, img_channels))
            # 给图片添加高斯噪声
            noisy_img = np.copy(image)
            noisy_img = noisy_img + gauss
            # 设置图片添加高斯噪声之后的像素值的范围
            noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
            attacked_frames.append(noisy_img)
            cv2.imwrite("Add_Gaussian.png", noisy_img)
        return attacked_frames
    @staticmethod
    def Add_Poisson(frame_list):
        attacked_frames=[]
        for image in frame_list:
            # 计算图像像素的分布范围
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            # 给图片添加泊松噪声
            noisy_img = np.copy(image)
            noisy_img = np.random.poisson(noisy_img * vals) / float(vals)
            attacked_frames.append(noisy_img)
            cv2.imwrite("Add_Poisson.png", noisy_img)
        return attacked_frames
    @staticmethod
    def MedianFilter(frame_list, kernel_size=3):
        """中值濾波"""
        attacked_frames=[]
        for image in frame_list:
            image = cv2.medianBlur(image, ksize=kernel_size)
            attacked_frames.append(image)
            cv2.imwrite("MedianFilter.png", image)
        return attacked_frames
    @staticmethod
    def MeanFilter(frame_list, kernel_size=3):
        """均值滤波"""
        attacked_frames=[]
        for image in frame_list:
            image = cv2.blur(image, ksize=(kernel_size, kernel_size))
            attacked_frames.append(image)

        return attacked_frames
