import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# files = glob.glob("dir-tsunami-data/dir-data4/*.4band.out")
# files = glob.glob("dir-tsunami-data/dir-data4/*_M8.0.4band.out")
# files = glob.glob("dir-tsunami-data/dir-data4/*_M[78]*.4band.out")
files = glob.glob("../datasets/tsunami/omit_2011/*")

tsunami_data = []

for file in files:
    print("file=",file)
    f=open(file,"r")
    data = f.readline()
    print("dummy data=",data)
    while True:
        data=f.readline()
        if data == '':
            break
        else:
            data3=data.split(",")
            #print("data3[4]=",data3[4])
            #print("data3[5]=",data3[5])
            #print("data3[6]=",data3[6])
            #print("data3[7]=",data3[7])
            #print("data3[8]=",data3[8])
            seismic_amp1=float(data3[4])
            seismic_amp2=float(data3[5])
            seismic_amp3=float(data3[6])
            # seismic_amp4=float(data3[7])
            tsunami_h=float(data3[7])
            data3_list=[seismic_amp1,seismic_amp2,seismic_amp3,\
                        tsunami_h]
            # print("data3_list=",data3_list)
            tsunami_data.append(data3_list)

print("num=",len(tsunami_data))

correct_data_t=[]
for t in tsunami_data:
    correct_data_t.append([t[3]])
    # correct_data_t.append([t[4],0.0])
    # print("c_d=",t[4])
print("n correct=",len(correct_data_t))

input_data_t=[]
for t in tsunami_data:
    input_data_t.append([t[0],t[1],t[2]])
    # input_data_t.append([t[2],t[3]])  # the same result
    # input_data_t.append([t[3]])  # the same result
    # print("i_d=",[t[0],t[1],t[2],t[3]])

# tsunami_ave=np.average(correct_data_t,axis=0)
# tsunami_std=np.std(correct_data_t-tsunami_ave)
#
# seis_ave=np.average(input_data_t,axis=0)
# seis_std=np.std(input_data_t-seis_ave)
#
# correct_data=(correct_data_t-tsunami_ave)/tsunami_std
# input_data=(input_data_t-seis_ave)/seis_std

tsunami_max=np.max(correct_data_t)
print("tsunami_max=",tsunami_max)
tsunami_min=np.min(correct_data_t)
# tsunami_min=0.0

seis_max=np.max(input_data_t)
seis_min=np.min(input_data_t)
# seis_min=0.0

correct_data=((correct_data_t-tsunami_min)/(tsunami_max-tsunami_min) * 2.0)-1
input_data=((input_data_t-seis_min) / (seis_max-seis_min) * 2.0)-1

# for i in range(len(correct_data)):
#   t_val =correct_data[i,0]
#   correct_data[i,0] = t_val
# correct_data[i,1] = -(t_val-1.0) -1.0

print("n input=",len(input_data))
print("input_data=",input_data)
print("correct_data=",correct_data)


#---------- 1-3 s --------------------------------------------------
plot_work=[]
for t in tsunami_data:
    plot_work.append(t[0])

# plt.scatter(plot_work,correct_data)
# plt.xlabel("seismic_amp 1-3s (cm)")
# plt.ylabel("tsunami")
# plt.grid(True)
# plt.show()
#---------- 3-9 s --------------------------------------------------
plot_work=[]
for t in tsunami_data:
    plot_work.append(t[1])

# plt.scatter(plot_work,correct_data)
# plt.xlabel("seismic_amp 3-9s (cm)")
# plt.ylabel("tsunami")
# plt.grid(True)
# plt.show()
#---------- 9-27 s --------------------------------------------------
plot_work=[]
for t in tsunami_data:
    plot_work.append(t[2])

# plt.scatter(plot_work,correct_data)
# plt.xlabel("seismic_amp 9-27s (cm)")
# plt.ylabel("tsunami")
# plt.grid(True)
# plt.show()
#---------- 27-81 s --------------------------------------------------
plot_work=[]
for t in tsunami_data:
    plot_work.append(t[3])

# plt.scatter(plot_work,correct_data)
# plt.xlabel("seismic_amp 27-81s (cm)")
# plt.ylabel("tsunami")
# plt.grid(True)
# plt.show()

#============================================================
index = list(range(len(input_data)))
# print('range=',range(len(input_data)))
# print('index=',index)
random.shuffle(index)

# index_train = index[:round(len(input_data)*8/10)]
index_train = index[:round(len(input_data)*5/10)]
# print(":round=",round(len(input_data)*8/10))
# print("index:=",index[:round(len(input_data)*8/10)])
# index_test = index[round(len(input_data)*8/10):]
index_test = index[round(len(input_data)*5/10):]
print("index:=",index[round(len(input_data)*8/10):])

input_data = np.array(input_data)
correct_data = np.array(correct_data)
print("shape correct_data=",np.shape(correct_data))

input_train = input_data[index_train , : ]
input_test = input_data[index_test , : ]
correct_train = correct_data[index_train]
correct_test = correct_data[index_test]

n_train = len(input_train)
n_test = len(input_test)
print("n_train=",n_train,"n_test=",n_test)

#============================================================
n_in = len(input_train[0]) # number of input layer
print("n_in=",n_in)
n_mid1 = 8       # middle layer
n_mid2 = 12       # middle layer
n_out = 1        # output layer
# n_out = 2        # output layer
wb_width = 0.1   # 
wb_width = 0.2   # 
# wb_width = 0.01  #  the same result
# wb_width = 5.2   #  the same result
eta = 0.01       # learning coefficient
eta = 0.01       # learning coefficient
epoch = 300
epoch = 2000
batch_size = 5
interval = epoch / 10

#============================================================
#class BaseLayer:
class MiddleLayer:
# class MiddleLayer(BaseLayer):
    def __init__(self ,n_upper ,n):
        self.w = wb_width * np.random.randn(n_upper ,n)
        self.b = wb_width * np.random.randn(n)
        # print("init baselayer");
    
    def update(self ,eta):
        # self.w -= eta * self.grad_w
        # self.b -= eta * self.grad_b
        corr_w = self.grad_w * eta
        corr_b = self.grad_b * eta
        #corr_w = self.grad_w      # the same result
        #corr_b = self.grad_b
        # for i1 in range(len(corr_w)):
        #    for i2 in range(len(corr_w[i1])):
        #       corr_w[i1,i2] *= eta
        # for i1 in range(len(corr_b)):
        #    corr_b[i1] *= eta
        # corr_w = eta * self.grad_w
        # corr_b = eta * self.grad_b
        #print("middle w update moto_w=",self.w,"corr_w=",corr_w,\
        #      "ato_w=",self.w + corr_w)
        #print("middle b update moto_b=",self.b,"corr_b=",corr_b,\
        #      "ato_b=",self.b + corr_b)
        self.w -= corr_w
        self.b -= corr_b

    def forward(self ,x):
        self.x = x
        u = np.dot(x ,self.w) + self.b
        self.y = 1 / (1 + np.exp(-u))
        # print("Middle Forward y=",self.y)
        # print("Middle Forward ")
    
    def backward(self ,grad_y):
        # print("ml-backward shape grad_y=",np.shape(grad_y))
        # print("ml-backward shape self.y=",np.shape(self.y))
        delta = grad_y * (1 - self.y) * self.y
        # delta = grad_y          # the same result
        # for i1 in range(len(grad_y)):
        #     for i2 in range(len(grad_y[i1])):
        #         delta[i1,i2] = grad_y[i1,i2] * \
        #                        (1 - self.y[i1,i2]) * self.y[i1,i2]
        # print("ml-backward shape delta=",np.shape(delta))
        # print("ml-backward shape self.x.T=",np.shape(self.x.T))
        self.grad_w = np.dot(self.x.T ,delta)   #  .T : transpose
        self.grad_b = np.sum(delta ,axis = 0)
        self.grad_x = np.dot(delta ,self.w.T)
        # print("ml-backward shape grad_w=",np.shape(self.grad_w))
        # print("ml-backward shape grad_b=",np.shape(self.grad_b))
        # print("ml-backward shape grad_x=",np.shape(self.grad_x))
        
        # print("Middle Back")

# class OutputLayer(BaseLayer):
class OutputLayer:
    def __init__(self ,n_upper ,n):
        self.w = wb_width * np.random.randn(n_upper ,n)
        self.b = wb_width * np.random.randn(n)
        # print("init baselayer");
    
    def update(self ,eta):
        # self.w -=eta * self.grad_w
        # self.b -= eta * self.grad_b
        corr_w = eta * self.grad_w
        corr_b = eta * self.grad_b
        # print("output w update moto_w=",self.w,"corr_w=",corr_w,\
        #       "ato_w=",self.w + corr_w)
        # print("output b update moto_b=",self.b,"corr_b=",corr_b,\
        #       "ato_b=",self.b + corr_b)
        self.w -= corr_w
        self.b -= corr_b

    def forward(self ,x):
        self.x = x
        # print("output w=",self.w)
        # print("output b=",self.b)
        u = np.dot(x ,self.w) + self.b
        self.y = u
        # print("Out Forward")

    def backward(self ,t):
    #       t: correct_train
        # print("Output backward")
        delta = (self.y - t) 
        #for k in range(len(t)):
            # delta[k] = (self.y[k] - t[k])  # * np.abs(t[k])
            #print("k=",k,"correct_train=",t[k],\
            #  "current=",self.y[k], " diff=",delta[k])
        # print("shape self.y=",np.shape(self.y ))
        # print("shape t=",np.shape(t ))
        self.grad_w = np.dot(self.x.T ,delta)
        # print("grad_w=",self.grad_w)
        # print("len grad_w=",len(self.grad_w))
        # print("w=",self.w)
        # print("type grad_w=",type(self.grad_w ))
        # print("size grad_w=",np.size(self.grad_w ))
        # print("shape grad_w=",np.shape(self.grad_w ))
        # print("type delta=",type(delta ))
        # print("size delta=",np.size(delta ))
        # print("size delta=",np.size(delta ))
        # print("shape delta=",np.shape(delta ))
        # print("shape self.w=",np.shape(self.w ))
        # sum delta -> correction
        self.grad_b = np.sum(delta ,axis=0)
        self.grad_x = np.dot(delta ,self.w.T)

#===================================================
middle_layer_1 = MiddleLayer(n_in ,n_mid1)
print("end of init middle layer 1")
# middle_layer_2 = MiddleLayer(n_mid1 ,n_mid2)
# print("end of init middle layer 2")
output_layer = OutputLayer(n_mid1 ,n_out)
print("end of init output layer")


#===========================================================
def forward_propagation(x):
    middle_layer_1.forward(x)                  # x:input
    # middle_layer_2.forward(middle_layer_1.y)   # y:output of sigmoid func. L1
    output_layer.forward(middle_layer_1.y)     # y:output of sigmoid func. L2

def backpropagation(t):
    # print("start backpropagation")
    output_layer.backward(t)                       # t:diff?
    # middle_layer_2.backward(output_layer.grad_x)   # grad of output
    # middle_layer_1.backward(middle_layer_1.grad_x) # grad of L.2
    middle_layer_1.backward(output_layer.grad_x) # grad of L.2

def update_wb(batch_size):
    middle_layer_1.update(eta/batch_size)
    # middle_layer_2.update(eta/batch_size)
    output_layer.update(eta/batch_size)

def get_error(t ,batch_size ):
    return 1.0 / 2.0 * np.sum(np.square(output_layer.y - t)) / batch_size

#===========================================================
train_error_x=[]
train_error_y=[]
test_error_x=[]
test_error_y=[]

n_batch = n_train // batch_size
print("n_batch=",n_batch)
for i in range(epoch):
    # print("i=",i,"epoch=",epoch)
    forward_propagation(input_train)
    error_train = get_error(correct_train ,n_train )
    # print("len(correct_train)=",len(correct_train))
    # for j in range(len(correct_train)):
    #    print("j=",j,"correct_train=",correct_train[j],\
    #          "predict=",output_layer.y[j], \
    #          " diff=",output_layer.y[j]-correct_train[j])
    forward_propagation(input_test)
    error_test = get_error(correct_test ,n_test )
    # print("error_train=",error_train,"error_test=",error_test)

    train_error_x.append(i)
    train_error_y.append(error_train)
    test_error_x.append(i)
    test_error_y.append(error_test)

    if i%interval==0:
        print("Epoch:"+str(i+1)+"/"+str(epoch),
            "Error_train:"+str(error_train),
            "Error_test:"+str(error_test))
    # -----------------------------------------------------------
    # 
    index_random=np.arange(n_train)
    np.random.shuffle(index_random)

    #
    for j in range(n_batch):
        # print("j=",j)
        mb_index = index_random[j*batch_size:(j+1)*batch_size]
        # print("mb_index =",mb_index )
        # print("shape mb_index =",np.shape(mb_index))
        x = input_train[mb_index , : ]
        # print("shape correct_train",np.shape(correct_train))
        # print("shape correct_train=",np.shape(correct_train))
        t = correct_train[mb_index]
        # print("in main shape t",np.shape(t))

        forward_propagation(x)
        backpropagation(t)
        update_wb(batch_size)


# ===========================================================
# 
plt.plot(train_error_x,train_error_y,label="Train")
plt.plot(test_error_x,test_error_y,label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()


# ===========================================================
#   used in M7 distinguish
forward_propagation(input_train)
# print("shape output_layer.y=",np.shape(output_layer.y))
# print("np.argmax(output_layer.y ,axis = 1)=",\
#        np.argmax(output_layer.y ,axis = 1))
# print("np.argmax(correct_train ,axis = 1)=",\
#        np.argmax(correct_train ,axis = 1))
#   argmax  index of maxvalue
count_train = np.sum(np.argmax(output_layer.y ,axis = 1) == \
                    np.argmax(correct_train ,axis = 1))
forward_propagation(input_test)
count_test = np.sum(np.argmax(output_layer.y ,axis = 1) == \
                    np.argmax(correct_test ,axis = 1))

print("Accuracy Train:" + str(count_train / n_train * 100)+"%" ,\
      "Accuracy Test:" + str(count_test / n_test * 100) + "%")


# ===========================================================
# 
# output_x=[]
# correct_y=[]
# 
# for i in range(len(correct_test )):
#    output_x.append([output_layer.y[i,0]])
#    correct_y.append([correct_test[i,0]])
output_x=(output_layer.y +1.0)/2.0 * (tsunami_max-tsunami_min) + tsunami_min
correct_y=(correct_test +1.0)/2.0 * (tsunami_max-tsunami_min) + tsunami_min

# output_x_t=[]
# correct_y_t=[]
# 
# for i in range(len(correct_test )):
#    output_x_t.append([output_layer.y[i,0]])
#    correct_y_t.append([correct_test[i,0]])
forward_propagation(input_train)
output_x_t=(output_layer.y +1.0)/2.0 * (tsunami_max-tsunami_min) + tsunami_min
correct_y_t=(correct_train +1.0)/2.0 * (tsunami_max-tsunami_min) + tsunami_min
# output_x=output_layer.y * tsunami_std + tsunami_ave
# correct_y=correct_test  * tsunami_std + tsunami_ave

plt.scatter(correct_y_t ,output_x_t,c="blue")
plt.scatter(correct_y ,output_x,c="red")
plt.ylabel("out put")
plt.xlabel("correct")
plt.grid(True)
plt.show()
