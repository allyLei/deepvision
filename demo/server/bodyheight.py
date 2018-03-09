#!/usr/local/bin/python2.7
# -*- coding:utf-8 -*-

import numpy as np
import math
import json

class HeightCalculation(object):

    def __init__(self, carema_type):
        conf_file = json.load(file('./conf/bodyheight.conf'))
        self.paramload = conf_file.has_key(carema_type)
        if not self.paramload:
            return
        configs = conf_file[carema_type]
        self.H = configs['measure']['H']
        self.L = configs['measure']['L']
        self.f_ = configs['measure']['f_']

        self.cameramatrix = configs['camera']['camera_matrix']
        self.distcoefs = configs['camera']['dist_coefs']
        self.w = configs['camera']['width'] # image width
        self.h = configs['camera']['height'] # iamge height

        self.mapx, self.mapy = self.init_map()

        self.angle1 = math.atan(self.L/self.H) # 成像最低角
        self.angle2 = math.atan(self.h/(self.f_*2)) # fov_h/2, calibrate

    # 初始化畸变矫正的映射矩阵
    def init_map(self):
        zeros = np.zeros([self.w, self.h])
        ones = np.ones(self.w * self.h,dtype=np.float32)
        u,v = np.where(~np.isnan(zeros))
        xyz = np.vstack(((u,v),ones))

        # 内参
        fx = self.cameramatrix[0][0]
        fy = self.cameramatrix[1][1]
        cx = self.cameramatrix[0][2]
        cy = self.cameramatrix[1][2]
        # 径向畸变
        k1 = self.distcoefs[0]
        k2 = self.distcoefs[1]
        k3 = self.distcoefs[4]

        invmtx = np.linalg.inv(self.cameramatrix)
        xyzc = np.dot(invmtx, xyz)

        r2 = np.square(xyzc[0,:]) + np.square(xyzc[1,:])
        x,y = xyzc[0,:], xyzc[1,:]
        
        # 畸变系数
        dist_val = (1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2)

        # (u, v) 对应的畸变坐标映射 (mapx, mapy)
        mapx = (fx * x * dist_val + cx).reshape([self.w,self.h])
        mapy = (fy * y * dist_val + cy).reshape([self.w,self.h])
        return mapx,mapy

    def DistortionCoordinateCorrection(self,point):
        se = np.square(self.mapx-point[0]) + np.square(self.mapy-point[1])
        ind = np.where(se == se.min())
        return (ind[0],ind[1])

    def correction(self, point_foot, point_top):
        point_foot = self.DistortionCoordinateCorrection(point_foot)
        point_top = self.DistortionCoordinateCorrection(point_top)
        if point_foot[1] < point_top[1]:
            point_foot, point_top = point_top, point_foot
        return point_foot, point_top

    def calculation(self, point_foot, point_top, imagesize=None):
        # Coordinate correction
        point_foot, point_top = self.correction(point_foot, point_top)
        # three angles about person
        angle3 = math.atan((self.h - 2*point_foot[1] )/(2*self.f_)) # 脚至水平中心线
        angle4 = math.atan((self.h - 2*point_top[1] )/(2*self.f_)) # 头至水平中心线
        angle5 = math.atan((self.w - 2*point_top[0] )/(2*self.f_)) # 头部点至横向中心线
        # 
        alpha = self.angle1+self.angle2+angle3 # 脚-相机-垂直
        beta = self.angle1+self.angle2+angle4 # 头顶-相机-垂直
        # 注：alpha>90时，说明人已经不在地面上，无法根据高度计算距离

        distance = self.H * math.tan(alpha) / math.cos(angle5)
        # 1.05 是根据正面站立时统计的经验值
        #height = self.H * math.sin(angle4-angle3) / ( math.cos(angle5) * math.cos(alpha) * math.sin(beta)) * 1.05
        height = 1.05 * distance * math.sin(angle4-angle3) / (math.sin(alpha) * math.sin(beta))
        return distance, height

    
    def calculationWithDistance(self, point_top, distance):

        angle3 = math.atan((self.h - 2*point_top[1] )/(2*self.f_)) # 头只中心线
        angle4 = math.atan((self.w - 2*point_top[0] )/(2*self.f_)) # 头部点至横向中心线

        alpha = self.angle1 + self.angle2 + angle3 # 头顶-相机-垂直线

        # 如果是distance是人到相机的距离，则不需要公式最后的除项
        height = self.H + (distance * math.tan(alpha - math.pi/2)) / math.cos(angle4) 
        return height

    def measure(self, box):
        distance,height = -1.0,-1.0
        # 读到参数并且接收到正确的坐标
        if self.paramload and box and len(box)>=4:
            x1 = box[0]*self.w
            y1 = box[1]*self.h
            x2 = box[2]*self.w
            y2 = box[3]*self.h
            middlex = (x1 + x2)/2
            point_top  = (middlex,y1)
            point_foot = (middlex,y2)
            distance,height = self.calculation(point_foot,point_top)
        return distance,height
