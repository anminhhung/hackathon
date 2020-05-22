import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gdown
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from annoy import AnnoyIndex
import zipfile
import time
import cv2
import os 
import shutil

# sroot_path = './'
data_df_ouput = pd.read_csv('data_df_output.csv')
data_df_output = data_df_ouput.reset_index()
if os.path.exists('static/output_images'):
    shutil.rmtree('static/output_images')

emblist = []
for line in data_df_ouput.embeddings:
    emb = []
    for ele in line.split(','):
       ele = ele.lstrip('[')
       ele = ele.rstrip(']')
       emb.append(float(ele))
    emblist.append(emb)

data_df_ouput['embedding'] = emblist

data_df_ouput.drop(columns=['Unnamed: 0', 'embeddings'], inplace=True)

# Using Spotify's Annoy
def get_similar_images_annoy(annoy_tree, img_index, number_of_items=12):
    start = time.time()
    img_id, img_label  = data_df_ouput.iloc[img_index, [0, 1]]
    similar_img_ids = annoy_tree.get_nns_by_item(img_index, number_of_items+1)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    # ignore first item as it is always target image
    return img_id, img_label, data_df_ouput.iloc[similar_img_ids[1:]] 


# for images similar to centroid 
def get_similar_images_annoy_centroid(annoy_tree, vector_value, number_of_items=5):
    start = time.time()
    similar_img_ids = annoy_tree.get_nns_by_vector(vector_value, number_of_items+1)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    # ignore first item as it is always target image
    return data_df_ouput.iloc[similar_img_ids[1:]] 


def show_similar_images(similar_images_df, fig_size=[10,10], hide_labels=True):
    if hide_labels:
        category_list = []
        for i in range(len(similar_images_df)):
            # replace category with blank so it wont show in display
            category_list.append(CategoryList(similar_images_df['label_id'].values*0,
                                              [''] * len(similar_images_df)).get(i))
    else:
        category_list = [learner.data.train_ds.y.reconstruct(y) for y in similar_images_df['label_id']]
    # return learner.data.show_xys([open_image(img_id) for img_id in similar_images_df['img_path']],
    #                             category_list, figsize=fig_size)
    return similar_images_df['img_path'].tolist()

print("building tree ....")
# more tree = better approximation
ntree = 100
#"angular", "euclidean", "manhattan", "hamming", or "dot"
metric_choice = 'angular'

annoy_tree = AnnoyIndex(len(data_df_ouput['embedding'][0]), metric=metric_choice)

# # takes a while to build the tree
for i, vector in enumerate(data_df_ouput['embedding']):
    annoy_tree.add_item(i, vector)
_  = annoy_tree.build(ntree)

print("finished build tree")

def centroid_embedding(outfit_embedding_list):
    number_of_outfits = outfit_embedding_list.shape[0]
    length_of_embedding = outfit_embedding_list.shape[1]
    centroid = []
    for i in range(length_of_embedding):
        centroid.append(np.sum(outfit_embedding_list[:, i])/number_of_outfits)
    return centroid

# dress
def recommend(img_path):
    img_path = './' + img_path
    outfit_img_ids = data_df_ouput[data_df_ouput['img_path'] == img_path].index[0]
    print("outfit_img_ids: ", outfit_img_ids)
    outfit_embedding_list = []
    #for img_index in outfit_img_ids:
    outfit_embedding_list.append(data_df_ouput.iloc[outfit_img_ids, 3])

    outfit_embedding_list = np.array(outfit_embedding_list)

    outfit_centroid_embedding = centroid_embedding(outfit_embedding_list)
    outfits_selected = data_df_ouput.iloc[outfit_img_ids] 

    similar_images_df = get_similar_images_annoy_centroid(annoy_tree, outfit_centroid_embedding)

    result = show_similar_images(similar_images_df, fig_size=[20,20])

    new_folder = "static/output_images"
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    my_path = []
    i = 1
    for image in result:
        image = image.split('/', 1)[-1]
        my_path.append(image)
        path = "static/output_images/" + str(i) + '.jpg'
        cv2.imwrite(path, cv2.imread(image))
        i += 1

    return my_path

