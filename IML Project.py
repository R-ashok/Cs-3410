import pandas as pd          
import numpy as np           
import matplotlib.pyplot as plt             
from mpl_toolkits.mplot3d import Axes3D                                              
from sklearn.decomposition import PCA #used only for one graph, can be omitted 
                              
data=pd.read_excel("games.csv.xlsx")         
                          
def NoLetters(x):   
    if not isinstance(x,str):                                                       
        return True                                   
    return all(not ch.isalpha() for ch in x)                                                                    
                                                                                         
invalid_white=data["white_id"].apply(NoLetters) 
invalid_black=data["black_id"].apply(NoLetters)                       
data=data[~(invalid_white|invalid_black)].reset_index(drop=True)                                 

#end of data cleaning    

#feature engineering                                                                          
data['created_at']= pd.to_datetime(data['created_at'],unit='ms',errors='coerce')            
data['last_move_at']=pd.to_datetime(data['last_move_at'],unit='ms',errors='coerce')         

data['game_length']=(data['last_move_at']- data['created_at']).dt.total_seconds()/60                                     
data=data[data['game_length']>0]                                 
     
data['RatingGap']=data['white_rating']- data['black_rating']                                            
data['rated10']=data['rated'].astype(int)                                  

# Split increment code into Time and Bonus (handle missing safely)                          
data[['Time','Bonus']]=data['increment_code'].str.split('+',expand=True)               
data['Time'] =pd.to_numeric(data['Time'],errors='coerce')                                           
data['Bonus']=pd.to_numeric(data['Bonus'],errors='coerce')                                          

data['opening_group']=data['opening_name'].apply(lambda x:' '.join(str(x).replace(':','').strip().split()[:2]))       

quantile_cap=data['game_length'].quantile(0.999)                               
data=data[data['game_length']<=quantile_cap].reset_index(drop=True)                 
data=data[data['game_length']>0].reset_index(drop=True)                                             

#data['avg_rating'] =(data['white_rating']+data['black_rating'])/2          
#median_rating=data['avg_rating'].median()          

data=data[data['Bonus']<150].reset_index(drop=True)             

features=['white_rating','black_rating','RatingGap','game_length','Time','Bonus','rated10','opening_ply']           
 
group_features=data.groupby('opening_group')[features].mean()                           
group_features_values=group_features.values 

means=np.mean(group_features_values,axis=0)#asix 0 denotes column wise mean      
stds =np.std(group_features_values, axis=0)#Standard Deviation          

stds[stds==0]=1e-200
group_X=(group_features_values-means)/stds                                                                               
                             
X=group_features.values                                                 
mean=X.mean(axis=0)           
std=X.std(axis=0)                           
group_X=(X-mean)/(std+1e-200)                                                                                    
                                         
#end of feature engineering                     

#K-Means Implementation                                 
def kmeans(X,k=6,max_iter=300,tol=1e-200):          
    n,d=X.shape                                                      

#Initialize centroids randomly from data points
    indices=np.random.choice(n,k,replace=False)       
    centroids=X[indices]  

    for iter in range(max_iter):    
        
#Assign each point to closest centroid  
        diff=X[:,None,:]-centroids[None,:,:] #(n_samples,1,n_features)- (1,k,n_features)=(n_samples,k,n_features)                        
        squared=diff**2                                 
        sum_squared=np.sum(squared,axis=2)                  
        distances=np.sqrt(sum_squared)           
        labels=np.argmin(distances,axis=1)                                                                                         

#Compute new centroids  
        new_centroids=np.zeros_like(centroids)                                               
        for j in range(k):                                                                               
            points=X[labels==j]                             
            if len(points)>0:                                            
                new_centroids[j]=points.mean(axis=0)                                                 
            else:                                                                                             
                new_centroids[j]=X[np.random.choice(n)] #reinitialize empty cluster randomly                        

#Check convergence(movement<tol)                                         
        dif=new_centroids-centroids         
        shift=np.sqrt(np.sum(dif**2))       
        if shift<tol:                                       
            print(f"Converged in {iter+1} iterations.")            
            break                

        centroids=new_centroids         
    return labels,centroids      

             
k=6                                                             
group_labels,centroids=kmeans(group_X, k=k)             
group_features['cluster']=group_labels           

#Map each game to its cluster                                                                       
opening_to_cluster=dict(zip(group_features.index,group_features['cluster']))     
data['cluster']=data['opening_group'].map(opening_to_cluster)  
 
#Remove NaNs(openings not in clusters)                                    
data_clustered=data.dropna(subset=['cluster']).reset_index(drop=True)    
y=data_clustered['cluster'].astype(int)                                              
X=data_clustered[features].fillna(0).values 
#standardize                   
X=(X-mean)/(std+1e-200) 
   
 
#train-test split                              
def train_test_split(X,y,test_size=0.2):    
    n=len(X)                                    
    indices=np.arange(n)          
    np.random.shuffle(indices)        # Shuffle data order
    
    test_count=int(n*test_size)         
    test_idx=indices[:test_count]           
    train_idx=indices[test_count:]                 
    X_train=X[train_idx]             
    X_test=X[test_idx]   
    y_train=y[train_idx]                
    y_test=y[test_idx]       
         
    return X_train,X_test,y_train,y_test                                                                        
          
                                                            
    

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)   

#Remove clusters that have fewer than 5 samples                                                        
cluster_counts=data_clustered['cluster'].value_counts() 
valid_clusters=cluster_counts[cluster_counts >=5].index         
data_clustered=data_clustered[data_clustered['cluster'].isin(valid_clusters)]       
  
#NN to predict cluster                  
num_classes=len(np.unique(y)) 
num_classes=y_train.max()+ 1                                                                                        
y_train_oh=np.zeros((y_train.size,num_classes))                     
y_train_oh[np.arange(y_train.size),y_train]= 1  
y_test_oh=np.zeros((y_test.size,num_classes))                    
y_test_oh[np.arange(y_test.size),y_test]=1                                   

#Helper functions                                            
def relu(x):               
    return np.maximum(0,x)       

def relu_derivative(x):                
    return (x>0).astype(float)  
  
def softmax(x):                                                             
    exp=np.exp(x-np.max(x,axis=1,keepdims=True))                                         
    return exp/np.sum(exp,axis=1,keepdims=True)  

def cross_entropy(pred,target):  
    return -np.mean(np.sum(target*np.log(pred+1e-200),axis=1))                                 
                                 
def accuracy(pred,target):                                                              
    return np.mean(np.argmax(pred,axis=1)==np.argmax(target,axis=1))         
             

#Model architecture 
input_dim=X_train.shape[1]                                           
hidden1=128         
hidden2=64                      
output_dim=num_classes               
                               
#Weight initialization                                                 
#np.random.seed(22)                                                                                       
W1=np.random.randn(input_dim,hidden1)*np.sqrt(2./input_dim)                            
b1=np.zeros((1,hidden1))                                                                     
W2=np.random.randn(hidden1,hidden2)*np.sqrt(2./hidden1)       
b2=np.zeros((1,hidden2))                
W3=np.random.randn(hidden2,output_dim)*np.sqrt(2./hidden2)                          
b3=np.zeros((1,output_dim))                                                          
                                                                                     
#Training parameters              
epochs=50                
lr=0.01                      
batch_size=32                        

                                                 
#Training loop            
                                                      
for epoch in range(epochs):                                                             
    idx=np.random.permutation(len(X_train))                          
    X_train=X_train[idx]                                    
    y_train_oh=y_train_oh[idx]                  
                                                                 
    for i in range(0,len(X_train),batch_size):        
        Xb=X_train[i:i+batch_size]              
        yb=y_train_oh[i:i+batch_size]                       
                                                             
        # Forward pass               
        z1=Xb@ W1+b1                         
        a1=relu(z1)   
        z2=a1@ W2+b2    
        a2=relu(z2)                                                 
        z3=a2@ W3+b3                                                                                     
        out=softmax(z3)                                                                                                                               
        #Compute loss                                                   
        loss=cross_entropy(out,yb)                                           
                                          
        #Backpropagation                
        dz3=(out-yb)/batch_size                
        dW3=a2.T@ dz3                
        db3=np.sum(dz3,axis=0,keepdims=True)
                    
        da2=dz3@ W3.T                                                           
        dz2=da2*relu_derivative(z2)             
        dW2=a1.T@ dz2                                                                 
        db2=np.sum(dz2,axis=0,keepdims=True)                                           
                                     
        da1=dz2@ W2.T                         
        dz1=da1*relu_derivative(z1)                           
        dW1=Xb.T@ dz1                                                                
        db1=np.sum(dz1,axis=0,keepdims=True)                                                              

        #Gradient descent update        
        W3-=lr*dW3           
        b3-=lr*db3      
        W2-=lr*dW2   
        b2-=lr*db2   
        W1-=lr*dW1    
        b1-=lr*db1             

    #Compute metrics every few epochs to display
    if (epoch+1)%5==0:                                                                 
        train_pred=softmax(relu(relu(X_train@ W1+b1)@W2 +b2) @W3+ b3)           
        test_pred=softmax(relu(relu(X_test@ W1+b1)@W2 +b2) @W3+ b3)     
        train_acc=accuracy(train_pred,y_train_oh)                                     
        test_acc=accuracy(test_pred,y_test_oh)                                                                                  
        print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | Loss: {loss:.4f}")   
     

#results                                                                     
train_pred=softmax(relu(relu(X_train@ W1+b1) @W2+b2) @W3+ b3)    
test_pred=softmax(relu(relu(X_test@ W1+b1) @W2+b2) @W3+ b3)                                               
train_acc=accuracy(train_pred, y_train_oh)                                                                                                                                                                          
test_acc=accuracy(test_pred, y_test_oh)

print(f"\nNN Training Accuracy (cluster): {train_acc:.3f}")                                    
print(f"NN Test Accuracy (cluster): {test_acc:.3f}")                     
                              
            
#Graphs for Visualisation                        

#Frequency of groups         
cluster_counts=data_clustered['opening_group'].value_counts().head(15)       
plt.figure(figsize=(8,6))                                                       
plt.barh(cluster_counts.index,cluster_counts.values,color='#228B22')    
plt.title("Most Common Opening Groups") 
plt.xlabel("Number of Games")    
plt.ylabel("Opening Group")       
plt.gca().invert_yaxis()    
plt.grid(alpha=0.3)  
plt.tight_layout()          
plt.show()   
#end of frequency of groups   

#PCA graph   
pca=PCA(n_components=2) #PCA done through package as it is only meant for visualisation 
X_pca=pca.fit_transform(group_X)

plt.figure(figsize=(8,6))    
for c in range(k): 
    plt.scatter(X_pca[group_labels==c,0],   
                X_pca[group_labels==c,1],   
                label=f"Cluster {c}",alpha=0.7)  
plt.title("K-Means Clusters") 
plt.xlabel("Principal Component 1")     
plt.ylabel("Principal Component 2") 
plt.legend()    
plt.grid(alpha=0.3)     
plt.show()        
#end of PCA graph       


#White vs Black rating by cluster
#   plt.figure(figsize=(8,6)) 
#   for c in range(k):                                                          
#       cluster_data=data_clustered[data_clustered['cluster']==c]                                                  
#       plt.scatter(cluster_data['white_rating'],cluster_data['black_rating'],s=20,label=f'Cluster {c}',alpha=0.6)   
#   plt.xlabel('White Rating')   
#   plt.ylabel('Black Rating')          
#   plt.title('White vs Black Rating by Cluster')    
#   plt.legend()            
#   plt.show()      
          
 
#For each numerical feature, plot of average and spread per cluster with more than 4 opening groups         
for feature in features:          
    plt.figure(figsize=(8,6))                                              
    cluster_means=data_clustered.groupby('cluster')[feature].mean()                        
    cluster_stds =data_clustered.groupby('cluster')[feature].std()                     

    plt.bar(cluster_means.index,cluster_means.values,yerr=cluster_stds.values,capsize=5,color='palevioletred',edgecolor='black')        
    plt.xlabel("Cluster")                 
    plt.ylabel(f"Average {feature}")                            
    plt.title(f"Cluster-wise Mean ± Std for {feature}")          
    plt.grid(alpha=0.3)          
    plt.tight_layout()   
    plt.show()        
#end of numerical feature graphs              

#some 2D graps pf kmeans by compenets
#   pairs=[("white_rating","black_rating"), ("white_rating","game_length"), ("game_length","opening_ply")]                                                                                                                                           
#   colors=plt.cm.tab10(np.linspace(0,1,len(data_clustered['cluster'].unique())))               
#                                                           
#   for(f1,f2) in pairs:    
#       plt.figure(figsize=(8,6))       
#       for c in sorted(data_clustered['cluster'].unique()):                  
#           subset = data_clustered[data_clustered['cluster']==c]  
#           plt.scatter(subset[f1],subset[f2],s=30,color=colors[c% len(colors)],label=f"Cluster {c}",alpha=0.6)          
#       plt.xlabel(f1)  
#       plt.ylabel(f2)                                                    
#       plt.title(f"{f1}vs{f2} by Cluster")     
#       plt.legend()         
#       plt.grid(alpha=0.3)         
#       plt.tight_layout()    
#       plt.show()   
#end of 2D plots     

#3D graph of kmeans by different components     
data_clustered['abs_rating_gap']=np.abs(data_clustered['RatingGap']) #simplified data for the sake of the graph
          
fig=plt.figure(figsize=(8,6))                         
ax =fig.add_subplot(111,projection='3d')                                                                               
x=data_clustered['white_rating']              
z=data_clustered['abs_rating_gap']
y=data_clustered['game_length']                      
c=data_clustered['cluster']                     

ax.scatter(x,y,z,c=c,cmap='tab10',s=40,alpha=0.8)                                                                                             
ax.set_xlabel('White Rating')         
ax.set_zlabel('Rating Gap')           
ax.set_ylabel('Game Length')                                                                                    
ax.set_title('3D Visualization of KMeans Clusters in Feature Space')                                    
plt.show()                                  
#graphs done