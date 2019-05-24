
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import pyperclip as pyclip

import mnist_inference
import mnist_train


strt=""

def evaluate(path):

    img=mpimg.imread(path)
    imgtemp=img[:,:,0]
    imgtemp=imgtemp.flatten()
    imgtemp=imgtemp/255.0
    imgtemp=np.mat(imgtemp)
    print(imgtemp)

    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y=mnist_inference.inference(x,None)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        if(ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess,ckpt.model_checkpoint_path)
        y=sess.run(y,feed_dict={x:imgtemp})

    o=np.array(y).flatten()

    print("判断结果：")
    print(o)

    count=0
    global strt
    strt="判断结果：\r\n"
    for i in o:
        strt+=str(count)
        strt+=':'
        strt+=str(i)
        strt+="\r\n"
        count+=1
    

    return np.argmax(o)


num=evaluate('./temp.jpg')
print("书写的数字是：%d"%num)
strt=strt+"\r\n书写的数字是：\r\n"
strt=strt+str(num)
pyclip.copy(strt)
