import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

#get feature
def transform(imgeFile):
    img = cv2.imread(imgeFile)
    resized_img = cv2.resize(img, (26, 26), interpolation=cv2.INTER_AREA)
    input_data = []
    for l in range(resized_img.shape[0]):
        hist = np.histogram(resized_img[l].T[2].flatten(), bins=20)[1].tolist()
        input_data += hist
    return input_data

#loading datas
def load_img(img_dir):
    images = []
    labels = []
    #print(img_dir)
    for root, dirs, files in os.walk(img_dir):
        for filename in (x for x in files if x.endswith('.png')):
            filepath = os.path.join(root, filename)
            #img = cv2.imread(filepath,cv2.IMREAD_COLOR)
            images.append(filepath)
            labels.append(filepath.split('/')[-2])
    return images,labels

#get data
def train_data(images,labels):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    train_idx,test_idx = train_test_split(range(len(y)),test_size=0.2,stratify = y, random_state = 4)
    train_y = y[train_idx]
    test_y = y[test_idx]
    x_rgb = np.row_stack([transform(img) for img in images])
    train_x = x_rgb[train_idx,:]
    test_x = x_rgb[test_idx,:]
    return train_x,train_y,test_x,test_y,label_encoder

#save model
def save_model(model,label_encoder,output_file):
    try:
        with open(output_file,'wb') as outfile:
            pickle.dump({
                'model':model,
                'label_encoder':label_encoder
            },outfile)
        return True
    except:
        return False

# acu
def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict_values[i]:
            correct += 1
    return correct / float(len(actual))

'''caculate'''
def eval_model(y_true,y_pred,labels):
    '''Precision，Recall，f1，support'''
    P,r,f1,s =precision_recall_fscore_support(y_true,y_pred)
    ''' Total Precision，Recall，f1，support'''
    tot_P = np.average(P,weights =s)
    tot_r = np.average(r,weights =s)
    tot_f1 = np.average(f1,weights =s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        'Label':labels,
        'Precision':P,
        'Reacll':r,
        'F1':f1,
        'Support':s
    })
    res2 = pd.DataFrame({
        'Label':['Total'],
        'Precision':[tot_P],
        'Recall':[tot_r],
        'F1':[tot_f1],
        'Support':[tot_s]
    })
    res2.index=[999]
    res = pd.concat([res1,res2])
    '''confusion matrix'''
    conf_mat = pd.DataFrame(confusion_matrix(y_true,y_pred),columns=labels,index=labels)
    return conf_mat,res[['Label','Precision','Recall','F1','Support']]


images = []
labels = []
print("loading image...")
#H
green_images,green_labels = load_img('/Users/peytonzhu/Desktop/RandomForest/GreenH/GreenH')
#hard
hard_images,hard_labels = load_img('/Users/peytonzhu/Desktop/RandomForest/hard-20210827T004414Z-001/hard')
#soft
soft_images,soft_labels = load_img('/Users/peytonzhu/Desktop/RandomForest/soft/soft')
#normal
normal_images,normal_labels = load_img('/Users/peytonzhu/Desktop/RandomForest/Normal/Normal')

images = green_images + hard_images + soft_images + normal_images
labels = green_labels + hard_labels + soft_labels + normal_labels
#print(labels)
train_x,train_y,test_x,test_y,label_encoder = train_data(images,labels)

#training
print("Training")
mode_rf = RandomForestClassifier(n_estimators =200)
mode_rf.fit(train_x,train_y) # Training data set
save_model(mode_rf,label_encoder,os.path.join("",'mode_rf.pkl'))
y_rf = mode_rf.predict(test_x)
'''model'''
conf_mat_lab_rf,evalues_rf = eval_model(test_y,y_rf,label_encoder.classes_)
print(conf_mat_lab_rf)
print(evalues_rf)
#testing
print("Testing")
rf_preds = mode_rf.predict(test_x)
labels = label_encoder.inverse_transform(rf_preds)
print("Random Forest Classifier",labels)
test_labels = label_encoder.inverse_transform(test_y)
print("Raw Data",test_labels)
print("Accuracy",accuracy(labels,test_labels))
#ROC
'''for i in range(len(test_y)):
    if test_y[i] == rf_preds[i]:
        rf_preds[i] = 1
    else:
        rf_preds[i] = 0
    test_y[i] = 1

false_positive_rate, true_positive_rate, _ = roc_curve(test_y,rf_preds)
roc_area_under_the_curve = auc(false_positive_rate, true_positive_rate)
lw=2
plt.figure(figsize=(6, 10))
plt.subplot(2, 1, 1)
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_area_under_the_curve)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for RF Clf')
plt.legend(loc="lower right")

plt.subplot(2, 1, 2)
confusion_matrix = pd.crosstab(test_y, rf_preds, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()'''
