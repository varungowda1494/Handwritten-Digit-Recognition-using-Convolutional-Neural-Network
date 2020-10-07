import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data      #input_data is used for loading the mnist data 
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)   


#Hyperparameters defined 
minibatch_size=128      #weights are updated after every minibatch
LR =0.001  
no_of_iterations=75000
input_size = 784    #28x28 input image
classes=10        #(0-9) digits
dropout=0.75


x=tf.placeholder(tf.float32,[None,input_size])   # "None" refers to the number of samples and input_size refers to the number of features in a sample
y=tf.placeholder(tf.float32,[None,classes])
keep_prob=tf.placeholder(tf.float32)


def convul(x,W,b,strides=1):                     #convul function is for applying convolution on the image
    res1=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    res2=tf.nn.bias_add(res1, b)
    return tf.nn.relu(res2)

def maxpooling(x, k=2):                          #maxpooling function is for applying max pooling on the resulting image
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')




def CNN(x, weights, biases, dropout):
    x=tf.reshape(x,shape=[-1,28,28,1])
    conv1=convul(x,weights['w1'],biases['b1'])    # convolutional layer1
    Max1=maxpooling(conv1)                        # maxpooling layer1
    
    conv2=convul(Max1,weights['w2'],biases['b2']) # convolutional layer2
    Max2=maxpooling(conv2)                        # maxpooling layer2
    
    
    full_con1=tf.reshape(Max2,[-1,weights['w3'].get_shape().as_list()[0]]) #Fully Connected Layer
    res1=tf.add(tf.matmul(full_con1,weights['w3']),biases['b3'])
    res2=tf.nn.relu(res1)
    
    res3=tf.nn.dropout(res2, dropout)   #applying dropout (its a kind regularization to avoid overfitting)
   
    output=tf.add(tf.matmul(res3, weights['out']), biases['out'])
    return output




weights={ 'w1': tf.Variable(tf.random_normal([5, 5, 1, 32])),'w2': tf.Variable(tf.random_normal([5, 5, 32, 64])),'w3': tf.Variable(tf.random_normal([7*7*64, 1024])),'out': tf.Variable(tf.random_normal([1024, classes]))}

biases={ 'b1': tf.Variable(tf.random_normal([32])),'b2': tf.Variable(tf.random_normal([64])),'b3': tf.Variable(tf.random_normal([1024])),'out': tf.Variable(tf.random_normal([classes]))}



pred=CNN(x,weights,biases,keep_prob)   #prediction
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels=y))  #cost calculation
optimizer=tf.train.RMSPropOptimizer(learning_rate=LR).minimize(cost)  #optimization using RMSProp optimizer

correct_pred=tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))



loss_l1=[]
iterations_l1=[]
accuracy_l1=[]
with tf.Session() as sess:
    init_var=tf.global_variables_initializer()
    sess.run(init_var)                               
    step=1
    print("STARTED TRAINING")
    while(step*minibatch_size<no_of_iterations):
        minibatch_xtrain, minibatch_ytrain=mnist.train.next_batch(minibatch_size)       # backpropogating and updating the weights for each minibatch 
        sess.run(optimizer,feed_dict={x:minibatch_xtrain, y:minibatch_ytrain,keep_prob:dropout})
        if (step%10==0):
            loss, acc=sess.run([cost,accuracy],feed_dict={x:minibatch_xtrain,y:minibatch_ytrain,keep_prob: 1.0})
            loss_l1.append(loss)
            iterations_l1.append(step*minibatch_size)      # displaying the accuracy for every 10 minibatches (for every 1280 iterations)
            accuracy_l1.append(acc)
            
            print("Iteration No: "+str(step*minibatch_size)+" minibatch loss: " + "{:.2f}".format(loss) + " Train data accuracy: " +"{:.3f}".format(acc))
        step += 1
    print("-------------------------------")
    print("Test data Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels,keep_prob: 1.0}))




#RELATED GRAPHS--------

plt.plot(iterations_l1,loss_l1)
plt.xlabel('iterations')
plt.ylabel('loss')                            # loss vs no of iterations graph for RMSProp Optimizer
plt.show()


plt.plot(iterations_l1,accuracy_l1)
plt.xlabel('iterations')                           # accuracy vs no of iterations for RMSProp Optimizer  
plt.ylabel('accuracy')
plt.show()

