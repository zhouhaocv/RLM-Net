import json
import scipy.io as scio
import os

with open(os.getcwd()+'/data/vrd/sg_dataset/sg_train_annotations.json','r') as f:
    datas = json.loads(f.read())
with open(os.getcwd()+'/data/vrd/json_dataset/annotations_train.json','r') as f:
    datas2 = json.loads(f.read())
with open(os.getcwd()+'/data/vrd/json_dataset/objects.json','r') as f:
    objects_sg = json.loads(f.read())
count = {}
for ii in objects_sg:
	count.update({ii: 0})
detetion_datas = []
for i in range(4000):
	data = datas[i]
	a={}
	a.update(filename=data['filename'])
	a.update(photo_id=data['photo_id'])
	a.update(height=data['height'])
	a.update(width=data['width'])
	a.update(objects_num=0)
	k=0
	objects = []
	object_pairs = []
	predicate_label = []
	filename = data['filename']
	# print(datas2[filename])
	for j in range(len(datas2[filename])):
		for object_type in ['subject','object']:
			subject = datas2[filename][j][object_type]
			b={}
			bbox=[]
			b.update(names=objects_sg[subject['category']])
			b.update(category_id=int(subject['category'])+1)
			bbox.append(subject['bbox'][2])
			bbox.append(subject['bbox'][0])
			bbox.append(subject['bbox'][3]-subject['bbox'][2]+1)
			bbox.append(subject['bbox'][1]-subject['bbox'][0]+1)
			if (subject['bbox'][3]-subject['bbox'][2]+1)<=1 or (subject['bbox'][1]-subject['bbox'][0]+1)<=1:
				print(subject)
			b.update(bbox=bbox)
			if b not in objects:
				objects.append(b)
				count[objects_sg[subject['category']]] = count[objects_sg[subject['category']]]+1
				k=k+1
	for j in range(len(datas2[filename])):				
		Subject = datas2[filename][j]['subject']
		b={}
		bbox=[]
		b.update(names=objects_sg[Subject['category']])
		b.update(category_id=int(Subject['category'])+1)
		bbox.append(Subject['bbox'][2])
		bbox.append(Subject['bbox'][0])
		bbox.append(Subject['bbox'][3]-Subject['bbox'][2]+1)
		bbox.append(Subject['bbox'][1]-Subject['bbox'][0]+1)
		b.update(bbox=bbox)

		Object = datas2[filename][j]['object']
		o={}
		bbox=[]
		o.update(names=objects_sg[Object['category']])
		o.update(category_id=int(Object['category'])+1)
		bbox.append(Object['bbox'][2])
		bbox.append(Object['bbox'][0])
		bbox.append(Object['bbox'][3]-Object['bbox'][2]+1)
		bbox.append(Object['bbox'][1]-Object['bbox'][0]+1)
		o.update(bbox=bbox)

		predicate = datas2[filename][j]['predicate']

		if [objects.index(b),objects.index(o)] not in object_pairs:
			object_pairs.append([objects.index(b),objects.index(o)])
			predicate_label.append(predicate)

	a.update(objects=objects)
	a.update(objects_num=k)
	a.update(objects_pairs=object_pairs)
	a.update(predicate_label=predicate_label)
	detetion_datas.append(a)
# print(detetion_datas)
print(count)
addpath = os.getcwd()+'/data/vrd/json_dataset/detection_annotations/'
filename_detection = 'instances_train_3.json'
if addpath:
	os.makedirs(addpath)

with open(addpath+filename_detection,'w')as file_obj:
	json.dump(detetion_datas,file_obj)


with open(os.getcwd()+'/data/vrd/sg_dataset/sg_test_annotations.json','r') as f:
    datas = json.loads(f.read())
with open(os.getcwd()+'/data/vrd/json_dataset/annotations_test.json','r') as f:
    datas2 = json.loads(f.read())
count = {}
for ii in objects_sg:
	count.update({ii: 0})
detetion_datas = []
for i in range(1000):
	data = datas[i]
	a={}
	if data['filename'] == '4392556686_44d71ff5a0_o.gif':
		data['filename'] = '4392556686_44d71ff5a0_o.jpg'
		print('true')
	a.update(filename=data['filename'])
	a.update(photo_id=data['photo_id'])
	a.update(height=data['height'])
	a.update(width=data['width'])
	a.update(objects_num=0)
	k=0
	objects = []
	object_pairs = []
	predicate_label = []
	filename = data['filename']
	# print(datas2[filename])
	for j in range(len(datas2[filename])):
		for object_type in ['subject','object']:
			subject = datas2[filename][j][object_type]
			b={}
			bbox=[]
			b.update(names=objects_sg[subject['category']])
			b.update(category_id=int(subject['category'])+1)
			bbox.append(subject['bbox'][2])
			bbox.append(subject['bbox'][0])
			bbox.append(subject['bbox'][3]-subject['bbox'][2]+1)
			bbox.append(subject['bbox'][1]-subject['bbox'][0]+1)
			b.update(bbox=bbox)
			if b not in objects:
				objects.append(b)
				count[objects_sg[subject['category']]] = count[objects_sg[subject['category']]]+1
				k=k+1
	for j in range(len(datas2[filename])):				
		Subject = datas2[filename][j]['subject']
		b={}
		bbox=[]
		b.update(names=objects_sg[Subject['category']])
		b.update(category_id=int(Subject['category'])+1)
		bbox.append(Subject['bbox'][2])
		bbox.append(Subject['bbox'][0])
		bbox.append(Subject['bbox'][3]-Subject['bbox'][2]+1)
		bbox.append(Subject['bbox'][1]-Subject['bbox'][0]+1)
		b.update(bbox=bbox)

		Object = datas2[filename][j]['object']
		o={}
		bbox=[]
		o.update(names=objects_sg[Object['category']])
		o.update(category_id=int(Object['category'])+1)
		bbox.append(Object['bbox'][2])
		bbox.append(Object['bbox'][0])
		bbox.append(Object['bbox'][3]-Object['bbox'][2]+1)
		bbox.append(Object['bbox'][1]-Object['bbox'][0]+1)
		o.update(bbox=bbox)

		predicate = datas2[filename][j]['predicate']
		predicate_label.append(predicate)
		object_pairs.append([objects.index(b),objects.index(o)])

	a.update(objects=objects)
	a.update(objects_num=k)
	a.update(objects_pairs=object_pairs)
	a.update(predicate_label=predicate_label)
	detetion_datas.append(a)
print(count)
addpath = os.getcwd()+'/data/vrd/json_dataset/detection_annotations/'
filename_detection = 'instances_test_3.json'
with open(addpath+filename_detection,'w')as file_obj:
	json.dump(detetion_datas,file_obj)
print(detetion_datas)
