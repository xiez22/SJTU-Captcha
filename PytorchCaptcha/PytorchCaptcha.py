import cv2
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time
import os

#Hyper Parameters
MIN_BIT_SIZE=10
BATCH_SIZE=32
EPOCH=100
LR=0.002

#样本生成器
sys.setrecursionlimit(4000)

#加载数据
captcha_data=None
captcha_label=None

try:
    print('尝试从上一次的训练中恢复...')
    captcha_data=np.load('./Data/data.npy')
    captcha_label=np.load('./Data/label.npy')
    print('已加载数据。')
except:
    print('数据加载失败。不过我们可以重新开始。')
    captcha_data=np.zeros((0,40,40))
    captcha_label=np.zeros((0))

def labeler():
    while(True):
        #随机生成网址进行训练
        print('正在从上海交通大学统一身份认证网站抓取Captcha数据，请稍后...')
        URL_PATH_SOURCE='%05d'%random.randint(1,99999)
        URL_PATH='https://jaccount.sjtu.edu.cn/jaccount/captcha?329841958306738.1280128764'+URL_PATH_SOURCE
        print(URL_PATH)

        #从SJTU网站获取数据
        cap=cv2.VideoCapture(URL_PATH)
        ret,img=cap.read()
        #二值化
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,img=cv2.threshold(img,210,255,cv2.THRESH_BINARY)
        print(img.shape)
        #cv2.imshow('Captcha',img)
        #cv2.waitKey(0)

        #从左至右进行划分
        #定义二维数组判断是否已经访问
        checkTemp=[[0 for i in range(100)] for j in range(40)]
        #DFS
        def checkSize(img,curPosx,curPosy):
            checkTemp[curPosx][curPosy]=1
            l=curPosy
            r=curPosy
            cnt=1
            dx=[0,1,0,-1]
            dy=[1,0,-1,0]
            for i in range(4):
                if (curPosx+dx[i] in range(40) and curPosy+dy[i] in range(100) and
                img[curPosx+dx[i]][curPosy+dy[i]]<255 and checkTemp[curPosx+dx[i]][curPosy+dy[i]]==0):
                    templ,tempr,num=checkSize(img,curPosx+dx[i],curPosy+dy[i])
                    l=templ if templ<l else l
                    r=tempr if tempr>r else r
                    cnt+=num
            return l,r,cnt

        #存储数据的数组
        new_data=np.zeros((0,40,40))
        new_label=np.zeros((0),dtype=int)

        for lpos in range(100):
            if np.mean(img[16:24,lpos])<200:
                l=r=cnt=0
                for i in range(16,24):
                    if img[i,lpos]<255:
                        l,r,cnt=checkSize(img,20,lpos)
                        break
                if cnt>MIN_BIT_SIZE and l-r<40:
                    #print(l,r,cnt)
                    lpos=r+5
                    mid=int((l+r)/2)
                    temp_img=np.ones((40,40),np.float)
                    temp_img[:,20-mid+l:20+r-mid+1]=img[:,l:r+1]/255
                    temp_img=temp_img[np.newaxis,:,:]
                    new_data=np.append(new_data,temp_img,axis=0)

        if(new_data.shape[0]!=4 and new_data.shape[0]!=5):
            print('抱歉，识别失败！\n正在重新开始。\n\n')
            continue

        print('识别完成，共有%d个字母。'%new_data.shape[0])
        plt.subplot(2,new_data.shape[0],1)
        plt.imshow(img)
        for i in range(new_data.shape[0]):
            plt.subplot(2,new_data.shape[0],new_data.shape[0]+i+1)
            plt.imshow(new_data[i],cmap='gray')
        plt.show()

        yes=input('识别是否正确？是（直接输入第一个字符)/否(0)/退出(1)：')
        if(yes=='0'):
            print('没事，我们可以重新开始！\n\n')
            continue
        elif(yes=='1'):
            break
        else:
            for i in range(new_data.shape[0]):
                if(i>0):
                    c=input('请输入第%d个字符：'%(i+1))
                else:
                    c=yes
                    print('第一个字符是',c)
                new_label=np.append(new_label,np.array([ord(c)-ord('a')],dtype=int),0)

            #数据的合成与保存
            captcha_data=np.append(captcha_data,new_data,0)
            captcha_label=np.append(captcha_label,new_label,0)
            np.save('E:/Test/Captcha/data.npy',captcha_data)
            np.save('E:/Test/Captcha/label.npy',captcha_label)
            print('好的！我们已经完成了共{}个数据的提取！\n\n'.format(captcha_label.shape[0]))

#卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.out=nn.Linear(5*5*64,26)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x).view(-1,5*5*64)
        out=self.out(x)
        return out

print('正在构建神经网络......')
dataset=Data.TensorDataset(
    torch.from_numpy(captcha_data).type(torch.FloatTensor).unsqueeze(1).cuda(),
    torch.from_numpy(captcha_label).type(torch.LongTensor).cuda()
    )

dataloader=Data.DataLoader(dataset,BATCH_SIZE,True)

cnn=CNN().cuda()
try:
    cnn.load_state_dict(torch.load('./Data/net_param.pkl'))
    print('已加载网络。')
except:
    print('加载网络失败！')

def train(show_result=False):
    print(cnn)

    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(cnn.parameters(),LR)

    print('开始训练...')
    #开始训练
    for epoch in range(EPOCH):
        for step,(batchx,batchy) in enumerate(dataloader):
            batchx.requires_grad=True
            prediction=cnn(batchx)
            loss=loss_func(prediction,batchy)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if step%10==0:
                print('EPOCH:',epoch,'Step:',step,'Loss:',loss.item())
                if(show_result):
                    prediction=torch.max(prediction,1)[1]
                    for i in range(16):
                        plt.subplot(4,4,i+1)
                        plt.imshow(batchx[i].squeeze().detach().cpu().numpy())
                        plt.title('{} {}'.format(chr(batchy[i].cpu().numpy()+ord('a')),
                                  chr(prediction[i].detach().cpu().numpy()+ord('a'))))
                    plt.show()

        torch.save(cnn.state_dict(),'E:/Test/Captcha/net_param.pkl')
        print('神经网络保存成功！')

cur_step=0
while(True):
    try:
        URL_PATH=input("请输入验证码的图片地址：")
        URL_PATH='https://jaccount.sjtu.edu.cn/jaccount/'+URL_PATH
        #随机生成网址进行训练
        print('正在从上海交通大学统一身份认证网站抓取Captcha数据，请稍后...')
        #URL_PATH_SOURCE='%05d'%random.randint(1,99999999999)
        #URL_PATH='https://jaccount.sjtu.edu.cn/jaccount/captcha?329841958306738.1280'+URL_PATH_SOURCE
        #print(URL_PATH)

        #从SJTU网站获取数据
        cap=cv2.VideoCapture(URL_PATH)
        ret,img=cap.read()
        img0=img
        #二值化
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,img=cv2.threshold(img,210,255,cv2.THRESH_BINARY)

        print("验证码提取完成！正在识别...")
        #从左至右进行划分
        #定义二维数组判断是否已经访问
        checkTemp=[[0 for i in range(100)] for j in range(40)]
        #DFS
        def checkSize(img,curPosx,curPosy):
            checkTemp[curPosx][curPosy]=1
            l=curPosy
            r=curPosy
            cnt=1
            dx=[0,1,0,-1]
            dy=[1,0,-1,0]
            for i in range(4):
                if (curPosx+dx[i] in range(40) and curPosy+dy[i] in range(100) and
                img[curPosx+dx[i]][curPosy+dy[i]]<255 and checkTemp[curPosx+dx[i]][curPosy+dy[i]]==0):
                    templ,tempr,num=checkSize(img,curPosx+dx[i],curPosy+dy[i])
                    l=templ if templ<l else l
                    r=tempr if tempr>r else r
                    cnt+=num
            return l,r,cnt

        #存储数据的数组
        new_data=np.zeros((0,40,40))
        new_label=np.zeros((0),dtype=int)

        lpos=0
        bflag=False
        while lpos<100:
                flag=False
                l=r=cnt=0
                #增大发现范围
                for i in range(5,35):
                    if img[i,lpos]<255:
                        l,r,cnt=checkSize(img,i,lpos)
                        if cnt>MIN_BIT_SIZE:
                            flag=True
                            break
                if(flag):
                    if cnt>MIN_BIT_SIZE and l-r<40:
                        #判断是否为异常字符
                        if(r-l>=18):
                            #print('遇到异常识别字符！已作切分处理！')
                            bflag=True
                            tempr=r
                            r=int((r+l)/2)
                            #print('长度为：',r-l)
                            mid=int((l+r)/2)
                            temp_img=np.ones((40,40),np.float)
                            temp_img[:,20-mid+l:20+r-mid+1]=img[:,l:r+1]/255
                            temp_img=temp_img[np.newaxis,:,:]
                            new_data=np.append(new_data,temp_img,axis=0)
                            l=r
                            r=tempr
                            #print('长度为：',r-l)
                            mid=int((l+r)/2)
                            temp_img=np.ones((40,40),np.float)
                            temp_img[:,20-mid+l:20+r-mid+1]=img[:,l:r+1]/255
                            temp_img=temp_img[np.newaxis,:,:]
                            new_data=np.append(new_data,temp_img,axis=0)
                            lpos=r+1
                        else:
                            #print('长度为：',r-l)
                            lpos=r+1
                            mid=int((l+r)/2)
                            temp_img=np.ones((40,40),np.float)
                            temp_img[:,20-mid+l:20+r-mid+1]=img[:,l:r+1]/255
                            temp_img=temp_img[np.newaxis,:,:]
                            new_data=np.append(new_data,temp_img,axis=0)
                lpos=lpos+1

        if(new_data.shape[0]<=1):
            plt.imshow(img)
            plt.show()
            continue
        #print('分割完成，共有%d个字母。'%new_data.shape[0])

        #居中
        temp_data=new_data.copy()
        letter_i=0
        for letter in 1-new_data:
            total_num=0
            x_sum=y_sum=0
            for i in range(40):
                for j in range(40):
                    y_sum+=i*letter[i][j]
                    #x_sum+=j*letter[i][j]
                    total_num+=letter[i][j]
            if int(y_sum/total_num)<20:
                for i in range(20-int(y_sum/total_num)):
                    new_data[letter_i,i,:]=1
                for i in range(20+int(y_sum/total_num)):
                    new_data[letter_i,i+20-int(y_sum/total_num)]=temp_data[letter_i,i].copy()
            elif int(y_sum/total_num)>20:
                for i in range(60-int(y_sum/total_num)):
                    new_data[letter_i,i]=temp_data[letter_i,i+int(y_sum/total_num)-20].copy()
                for i in range(60-int(y_sum/total_num),40):
                    new_data[letter_i,i]=1
            letter_i+=1

        
        
        #通过神经网络识别
        prediction=torch.softmax(cnn(torch.from_numpy(new_data).type(torch.FloatTensor).unsqueeze(1).cuda()),1)
        chance=torch.max(prediction,1)[0].cpu().detach().numpy()
        result=torch.max(prediction,1)[1].cpu().detach().numpy()
        result_str=''
        for i in result:
            result_str+=chr(ord('a')+i)
        os.system('cls')
        print("识别的结果是{}".format(result_str))
        #print("该验证码的网址是:",URL_PATH)
        plt.subplot(2,new_data.shape[0],1)
        plt.imshow(img0)
        plt.title('Prediction:{}'.format(result_str))
        for i in range(new_data.shape[0]):
            plt.subplot(2,new_data.shape[0],new_data.shape[0]+i+1)
            plt.imshow(new_data[i],cmap='gray')
            plt.title('%.4f'%chance[i])
        plt.show()
    except:
        print("出现错误！！！正在重新开始...")
        time.sleep(3)