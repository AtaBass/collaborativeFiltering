import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from numpy import loadtxt


###### assistant methods ######

def load_precalc_params_small():

    file = open('./data/small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./data/small_movies_W.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_b.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./data/small_movies_Y.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_R.csv', 'rb')
    R = loadtxt(file,delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    #returns df with and index of movies in the order they are in in the Y matrix
    df = pd.read_csv('./data/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)


def normalizeRatings(Y, R):
    
    #process data by subtracting mean rating for every movie (every row).
    #only if  include real ratings R(i,j)=1.
    
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)



#################################
###### collaborative filtering ######

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y: ", Y.shape, "R: ", R.shape)
print("X: ", X.shape)
print("W: ", W.shape)
print("b: ", b.shape)
print("number of features: ", num_features)
print("number of movies: ",   num_movies)
print("number of users: ",    num_users)

tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"mean rating for movie 1 : {tsmean:0.3f} / 5" )

#cost for the content-based filtering
def cofi_cost_func(X,W,b,Y,R, lambda_):
    nm,nu = Y.shape
    J = 0

    for j in range(nu):
        w = W[j,:]
        b_j = b[0,j]
        for i in range(nm):
            x = X[i,:]
            y = Y[i,j]
            r = R[i,j]
            J += np.square(r * (np.dot(w,x) + b_j - y ) )
        
        
    J = (J/2) + (lambda_/2) * (np.sum(np.square(W)) + np.sum(np.square(X)))   
    return J        

#reduce the data set size for fast run
num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

#cost function
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0)
print(f"Cost: {J:0.2f}")

#cost function with regularization 
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5)
print(f"Cost (with regularization): {J:0.2f}")


#vectorized for speed
#uses tensorflow
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
   
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

#cost func
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")

#cost function with regularization 
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")




###############################

movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies) #init my ratings         

my_ratings[2700] = 5 
my_ratings[2609] = 2
my_ratings[929]  = 5   
my_ratings[246]  = 5   
my_ratings[2716] = 3   
my_ratings[1150] = 5   
my_ratings[382]  = 2   
my_ratings[366]  = 5   
my_ratings[622]  = 5   
my_ratings[988]  = 3   
my_ratings[2925] = 1   
my_ratings[2937] = 1   
my_ratings[793]  = 5  
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\n new user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'rated {my_ratings[i]} ->  {movieList_df.loc[i,"title"]}')


#load ratings
Y, R = load_ratings_small()
#add new user ratings to Y 
Y = np.c_[my_ratings, Y]
#add new user indicator matrix to R
R = np.c_[(my_ratings != 0).astype(int), R]

#subtract the mean = normalize the dataset
Ynorm, Ymean = normalizeRatings(Y, R)



#useful values
num_movies, num_users = Y.shape
num_features = 100


tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

#instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)



#run the algorithm

iterations = 200
lambda_ = 1
for iter in range(iterations):
    
    with tf.GradientTape() as tape:

        #compute the cost -> J
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    #derivative for J
    grads = tape.gradient( cost_value, [X,W,b] )

    
    #to minimize the loss -> iterations
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    #periodic output
    if iter % 20 == 0:
        print(f"training loss -> iteration {iter}: {cost_value:0.1f}")



#make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

#sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'predicting rating : {my_predictions[j]:0.2f} -> for movie {movieList[j]}')

print('\n\noriginal vs predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')


filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)


