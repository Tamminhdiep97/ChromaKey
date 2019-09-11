import numpy as np
import cv2 
import time
from sklearn.cluster import MeanShift
import threading

def init_1d_Matrix(img):  #Init [row, col, R,G,B]
    p=[]
    rows,cols, dimension = img.shape
    print(rows,cols)
    count=0
    for i in range(rows):
        for j in range(cols):
            k = [i,j,img[i,j][0],img[i,j][1],img[i,j][2]]
            p.insert(count,k)
            count+=1
    return p

def init_1d_Matrix_No_xy(a): #Init [i]=[R,G,B]
    a_1=[]
    for i in range (0,len(a)):
        a_1.append([a[i][2],a[i][3],a[i][4]])
    return a_1

def init_2d_Matrix(img): #Init [row][col]=[R,G,B]

    rows,cols,dimension = img.shape
    p=[]

    for i in range(rows):
        p.append([0]*cols)
        for j in range(cols):
            k=[img[i,j][0],img[i,j][1],img[i,j][2]]
            p[i][j]=k
    return p

def Mean_shift_function(Matrix):
    threading.current_thread().name = 'MainThread'
    cluster=MeanShift(bin_seeding=True, n_jobs =-1,).fit(Matrix)

    labels = cluster.labels_
    cluster_centers = cluster.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    return [cluster,n_clusters_,cluster_centers]  #[result, number_of_cluster, cluster_center]

def make_mapArray(a,a_1): #a_mean_result, a_1 is 1d_Matrix with x,y   #map_label_position[i]=[position] position of 1d_matrix which have the same cluster i

    p=a[0].labels_  #a[0] is a matrix, map from position_to_label
    map_label_position=[]
    for i in range(0,a[1]):
        map_label_position.append([])

    for i in range(0,len(a_1)):
        for j in range(0,a[1]):
            if p[i]==j:
                map_label_position[j].append(i)
    return map_label_position

def mean_predict(cor_,array,a2): #cordinate, meanshift_result
    k=[]
    for i in range(0,len(cor_)):
        tem=[a2[cor_[i][0]][cor_[i][1]][0], a2[cor_[i][0]][cor_[i][1]][1], a2[cor_[i][0]][cor_[i][1]][2] ] #a2[row][col][0], a2[row][col][1], a2[row][col][2]
        k.append(tem)
    h = []
    for i in range (0,len(cor_)):
        l=np.array([k[i]])
        h.append(array[0].predict(l)[0])
    h_unique = np.unique(h)
    return h_unique

def prepairArray(a,Array1d, Array2d,Map,img,img_blend): # a is [] with values are cluster, Map is map from cluster_postion, img_original, img_blend
    p=Array1d
    q=Array2d

    rows,cols,dimension = img.shape
    img_return=img
    for j in range(0,len(a)): #loop throught each cluster
        for i in range(0,len(Map[a[j]])): #a[j] <=> cluster => Map[a[j]] = Map[cluster] #loop throuhgt each element in each cluster
            c=p[Map[a[j]][i]][0]    # Map[a[j]][i]= position, p[  Map[a[j]][i]   ][0] <=> 1d[position][0] = row 
            d=p[Map[a[j]][i]][1]    # Map[a[j]][i]= position, p[  Map[a[j]][i]   ][1] <=> 1d[position][1] = col
            img_return[c,d]=img_blend[c,d] #change value of Original with Background
    return img_return

#main program

print("Input name of Orginal img: ")
original = input()
print("Input name of Background img")
background = input()

img1 = cv2.cv2.imread(original,cv2.cv2.IMREAD_COLOR) #import the img
img2 = cv2.cv2.imread(background,cv2.cv2.IMREAD_COLOR) #import the img

p_1=init_1d_Matrix(img1)    #make 1d matrix for meanshift
p_2=init_2d_Matrix(img1)    #make 2d matrix for img recontruction
p_3=init_1d_Matrix_No_xy(p_1)

t1=time.time()              #time start


mean_result=Mean_shift_function(p_3) #meanShift Img

t2=time.time() #stop time
t=int(t2-t1)
print("Time: ", t)

#Choose point
print("Input Number of point chosen:")
n=int(input())
cordinate=[]
for i in range(0,n):
    cordinate.append([0,0])
    print("x (col): ")
    x=int(input())
    print("y (row): ")
    y=int(input())
    cordinate[i]=[y,x]

#predict points chosen
a = mean_predict(cordinate,mean_result,p_2)  #predict x,y
print("cluseter predict: ",a)


#Make Map from cluster to postion in 1d Matrix
MapArray=make_mapArray(mean_result,p_1) #make mapArray



#Change Value in original img
result2d=prepairArray(a,p_1,p_2,MapArray,img1,img2)





cv2.cv2.imshow("Simple_black", result2d)
cv2.cv2.waitKey(0)


