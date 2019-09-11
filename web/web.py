from flask import Flask, render_template, request, send_from_directory, redirect
import glob, os
from werkzeug.utils import secure_filename
import numpy as np
import argparse
import time
import cv2
import os
import re
import threading
from sklearn.cluster import MeanShift
from flask import flash


#meanShift function
def init_1d_Matrix(img):
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

def init_1d_Matrix_No_xy(a):
    a_1=[]
    for i in range (0,len(a)):
        a_1.append([a[i][2],a[i][3],a[i][4]])
    return a_1

def init_2d_Matrix(img):

    rows,cols,dimension = img.shape
    p=[]

    for i in range(rows):
        p.append([0]*cols)
        for j in range(cols):
            k=[img[i,j][0],img[i,j][1],img[i,j][2]]
            p[i][j]=k
    return p

def Mean_shift_function(Matrix):
    cluster=MeanShift(bin_seeding=True, n_jobs =-1,).fit(Matrix)
    labels = cluster.labels_
    cluster_centers = cluster.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return [cluster,n_clusters_,cluster_centers] #result, number_of_Cluster, center

def make_mapArray(a,a_1):    #map_label_position[i]=[position] position which have the same cluster i

    p=a[0].labels_
    map_label_position=[]
    for i in range(0,a[1]):
        map_label_position.append([])

    for i in range(0,len(a_1)):
        for j in range(0,a[1]):
            if p[i]==j:
                map_label_position[j].append(i)
    return map_label_position

def mean_predict(cor_,array,p_2): #cordinate, meanshift_result
    k=[]
    for i in range(0,len(cor_)):
        tem=[p_2[cor_[i][0]][cor_[i][1]][0], p_2[cor_[i][0]][cor_[i][1]][1], p_2[cor_[i][0]][cor_[i][1]][2] ] #p_2[row][col][0], p_2[row][col][1], p_2[row][col][2]
        k.append(tem)
    h = []
    for i in range (0,len(cor_)):
        l=np.array([k[i]])
        h.append(array[0].predict(l)[0])
    h_unique = np.unique(h)
    return h_unique

def prepairArray(a,Array1d, Array2d,Map,img,img_blend):
    p=Array1d
    q=Array2d

    rows,cols,dimension = img.shape
    img_return=img
    for j in range(0,len(a)):
        for i in range(0,len(Map[a[j]])): #a[j] <=> cluster => Map[a[j]] = Map[cluster]
            c=p[Map[a[j]][i]][0]
            d=p[Map[a[j]][i]][1]
            img_return[c,d]=img_blend[c,d]
    return img_return


#webFunction
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

DEBUG = True

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/image-search", methods=['POST'])
def image_search():
    threading.current_thread().name = 'MainThread'
    #Get data from form
    if request.method == 'POST':
        if 'file_original' not in request.files and 'file_background' not in request.files:
            flash('Imgs type wrong')
            return redirect(request.url)

        file1 = request.files['file_original']
        if file1.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file1 and allowed_file(file1.filename):
            filename1 = secure_filename(file1.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))

        file2 = request.files['file_background']
        if file2.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file2 and allowed_file(file2.filename):
            filename2 = secure_filename(file2.filename)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))


        file3 = int(request.form['rol1']) 
        file4 = int(request.form['col1'])
        file5 = int(request.form['rol2'])
        file6 = int(request.form['col2'])

        #End: Get data from form

        img1 = cv2.imread(os.path.sep.join(["static/uploads/", filename1])) #load img_Original
        #cv2.cv2.imshow("Simple_black", image)
        #cv2.cv2.waitKey(0)
        img2 = cv2.imread(os.path.sep.join(["static/uploads/", filename2])) #load img_background
        #cv2.cv2.imshow("Simple_black", image)
        #cv2.cv2.waitKey(0)
        
       

        p_1=init_1d_Matrix(img1)    #make 1d matrix [i]=[rol,col,b,r,g]
        p_2=init_2d_Matrix(img1)    #make 2d matrix for img recontruction [rol][col] = [b,r,g]
        p_3=init_1d_Matrix_No_xy(p_1) #make 1d matrix for meanshift [i]=[b,r,g]
       
        t1=time.time()              #time start

        mean_result=Mean_shift_function(p_3) #meanShift Img_Original
        
        cordinate=[]
        for i in range(0,2):
            cordinate.append([0,0])

        cordinate[0]=[file3,file4]
        cordinate[1]=[file5,file6]

        a = mean_predict(cordinate,mean_result,p_2)  #predict x,y

        t2=time.time() #stop time
        t3=int(t2)

        MapArray=make_mapArray(mean_result,p_1) #make mapArray

        result2d=prepairArray(a,p_1,p_2,MapArray,img1,img2) #blend ImgArray
        cv2.imwrite(os.path.sep.join(["static/images/", "result"+str(t3)+".jpg"]),result2d)


#print(result2d)
       
        print(int(t2-t1))

        path=[]
        files=[]
        path.append(os.path.sep.join(["static/images", "result"+str(t3)+".jpg" ]))
        files.append(glob.glob(path[0]))
        return render_template('image-search.html', files=files, path=path)
    


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images/', path)

if __name__ == "__main__":
	app.run()
