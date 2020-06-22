import numpy as np
import cv2
import json
import os
import scipy.io as io
##read datasets
with open(os.getcwd()+'/data/vrd/json_dataset/annotations_train.json','r') as f:
    datas = json.loads(f.read())
with open(os.getcwd()+'/data/vrd/json_dataset/predicates.json','r') as f:
    predicates_sg = json.loads(f.read())
num_preds = len(predicates_sg)

#generate spatial anchors
spatial_anchors=np.zeros([num_preds,224,224,3]) # size: num_preds*height*width*3
count = np.zeros(num_preds) #size: num_preds
k=0
for i in datas:
	img = cv2.imread(os.getcwd()+'/data/vrd/sg_dataset/sg_train_images/'+i) # read image
	sp = img.shape
	data = datas[i]
	for j in range(len(data)):
		sub_box = data[j]['subject']['bbox'] #[ys,ym,xs,xm]
		obj_box = data[j]['object']['bbox']#[ys,ym,xs,xm]
		pred_label = data[j]['predicate']
		#relationship union region
		xs = min(sub_box[2],obj_box[2])
		ys = min(sub_box[0],obj_box[0])
		xm = max(sub_box[3],obj_box[3])
		ym = max(sub_box[1],obj_box[1])

		a=np.zeros([sp[0],sp[1],3])
		a[sub_box[0]:sub_box[1],sub_box[2]:sub_box[3],0] = 1
		a[obj_box[0]:obj_box[1],obj_box[2]:obj_box[3],2] = 1
		a[sub_box[0]:sub_box[1],sub_box[2]:sub_box[3],1] = 0.5
		a[obj_box[0]:obj_box[1],obj_box[2]:obj_box[3],1] += 0.5
	
		a = a[ys:ym,xs:xm]
		a = cv2.resize(a,(224,224))
		spatial_anchors[pred_label,:,:,:] += a
		count[pred_label] += 1
	k +=1
	print('have processed',k,'imgs')
datapath = os.getcwd()+'/data/spatial_anchors/' # save path
if not os.path.exists(datapath):
	os.mkdir(datapath)
for i in range(num_preds):
	spatial_anchors[i,:,:,:] = (255*spatial_anchors[i,:,:,:])/count[i]
	word = predicates_sg[i]
	cv2.imwrite(datapath+word+'.jpg',spatial_anchors[i,:,:,:])
np.save(os.getcwd()+'/data/spatial_anchors.npy',spatial_anchors)

#generate A_sim
A_sim=np.zeros([num_preds,num_preds])
for i in range(num_preds):
	img1_s = spatial_anchors[i,:,:,0]
	img1_o = spatial_anchors[i,:,:,2]
	for j in range(num_preds):
		img2_s = spatial_anchors[j,:,:,0]
		img2_o = spatial_anchors[j,:,:,2]
		nums = (np.sum((img1_s-img2_s)**2)+np.sum((img1_o-img2_o)**2))/(224*224*2);
		if nums<1000: #thresh
			A_sim[i,j]=1;

io.savemat(os.getcwd()+'/data/A_sim2.mat',{'A_sim':A_sim})
