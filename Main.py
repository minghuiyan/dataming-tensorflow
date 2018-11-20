import csv
import tensorflow as tf
batchsize=500
totalsize=743991
batchNo=totalsize//batchsize
filepath = "C:\\Users\\456\\Desktop\\gamedata\\train_all.csv" # 定义文件名称，本文件要与当前的.py文件要在同一文件夹下，不然要用绝对路径
testpath="C:\\Users\\456\\Desktop\\gamedata\\republish_test.csv"
submitpath="C:\\Users\\456\\Desktop\\gamedata\\submit.csv"
shot_dic={99999825:0,90063345:1,90109916:2,89950166:3,89950168:4,89950167:5,90155946:6,99999828:7,99999826:8,99999827:9,99999830:10}
converse_shot_dict = {v : k for k, v in shot_dic.items()}
def parse_to_one_shot(x):#将数字转化为标签向量
    s=shot_dic[x]
    m=[]
    for i in range(11):
        if i==s:
            m.append(1.0)
        else:
            m.append(0.)
    return m
def convert_vector_to_int(vector):
    max=0
    for i in range(11):
        if(vector[i]>vector[max]):
            max=i
    return converse_shot_dict[max]
def readTrainFile(filepath,start,batchsize):
    with open(filepath, 'r') as csvfile:  # 打开数据文件
        reader = csv.reader(csvfile)  # 用csv的reader函数读取数据文件,返回一个生成器
        x_data = []  # 定义一个空数组用于保存输入的数据
        y_data = []  #定义一个空数组用于保存标签数据
        for index, line in enumerate(reader):  # 循环读取数据文件并保存到数组data中
            if (index < start):
                continue
            elif (index > start + batchsize):
                break
            line1 = []
            i = 0
            for x in line: # line是个一维数组，是数据文件中的一行数据
                i = i + 1
                if (i == 26):
                    y_data.append(parse_to_one_shot(int(x)))
                elif (i == 27):
                    continue
                else:
                    try:
                        line1.append(float(x))
                    except ValueError as ve:
                        line1.append(0.)
            x_data.append(line1)
        #m=tf.constant(x_data)
        #n=tf.constant(y_data)
        return x_data,y_data
def write_to_csv(filepath,userid,service):
    with open(filepath,'a+',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([userid,service])
def add_layer(inputs,inputsize,outputsize,activation_function=None):
    Weights=tf.Variable(tf.random_normal([inputsize,outputsize]))
    Biases=tf.Variable(tf.zeros([1,outputsize]))
    Wx_plus_b=tf.matmul(inputs,Weights)+Biases
    if activation_function==None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
def main():

    x = tf.placeholder(tf.float32, [None, 25])
    y = tf.placeholder(tf.float32, [None, 11])

    # 构建输入层，其实就是设定权值
    L1=add_layer(x,25,10,tf.nn.sigmoid)
    # 构建中间层
    L2 = add_layer(L1,10,10,tf.nn.sigmoid)
    # 构建输出层
    predicition = add_layer(L2,10,11,tf.nn.softmax)
    # 损失函数
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predicition)
    # 使用梯度下降法训练，实际上就是求局部最优解
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # 初始化变量
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(5):
            for batch in range(batchNo):
                start=batch*batchsize+1
                #start=140400
                print(step,start)
                x_data,y_data=readTrainFile(filepath=filepath,start=start,batchsize=batchsize)
                sess.run(train_step, feed_dict={y: y_data, x: x_data})

        # 获得预测值
        with open(testpath, 'r') as csvfile:  # 打开数据文件
            reader = csv.reader(csvfile)  # 用csv的reader函数读取数据文件,返回一个生成器
            isFirst=True
            for index, line in enumerate(reader):  # 循环读取数据文件并保存到数组data中
                if (index < 1):
                    continue
                test_data = [] #包装一下
                data=[]#读取一行生成的向量
                i = 0
                userid=""
                for s in line: # line是个一维数组，是数据文件中的一行数据
                    i = i + 1
                    if (i == 26):
                        userid=s
                    else:
                        try:
                            data.append(float(s))
                        except ValueError as ve:
                            data.append(0.)
                test_data.append(data)
                pred_vector=sess.run(predicition,{x:test_data})
                pred_service=convert_vector_to_int(pred_vector[0])
                if(isFirst):#是不是第一次写
                    write_to_csv(submitpath, "userid", "current_service")  # 写标题
                    write_to_csv(submitpath,userid,pred_service)#写数据
                    isFirst=False
                else:
                    write_to_csv(submitpath,userid,pred_service)

main()