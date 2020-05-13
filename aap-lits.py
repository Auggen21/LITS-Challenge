# -*- coding: utf-8 -*-
"""lits_out.ipynb

"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from tensorflow.keras.models import model_from_json
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import time


json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")

input_shape = [64, 64, 1]

def slice_to_patch(slice, patch_ratio):
  
  slice[slice == 1] = 0
  slice[slice == 2] = 1
  
  patch_list = []
  
  for x_bin in range(2, len(patch_ratio)):
    for y_bin in range(2, len(patch_ratio)):
      patch = slice[patch_ratio[x_bin-2] : patch_ratio[x_bin], patch_ratio[y_bin - 2] : patch_ratio[y_bin]]
      patch = patch.reshape(patch.shape + (1,))
      patch_list.append(patch)
  
  return np.array(patch_list)

def patch_to_slice(patch, patch_ratio, input_shape, conf_threshold):
  
  slice = np.zeros((512, 512, 1))
  row_idx = 0
  col_idx = 0
  
  for i in range(len(patch)):
    
    slice[patch_ratio[row_idx]:patch_ratio[row_idx + 2], patch_ratio[col_idx]:patch_ratio[col_idx + 2]][patch[i] > conf_threshold] = 1
    
    col_idx += 1
    
    if i != 0 and (i+1) % 15 == 0:
      row_idx += 1
      col_idx = 0
  
  return slice

Builder.load_string("""
<MyWidget>:
    id: my_widget
    FileChooserListView:
        id: filechooser
        on_selection: my_widget.selected(filechooser.selection)  
    Button
        text: "open"
        on_release: my_widget.open(filechooser.path, filechooser.selection)            
""")



class MyWidget(BoxLayout):
    

    def open(self, path, filename):        
        img_path=os.path.join(path, filename[0])
        img_ex = nib.load(img_path).get_data()
        r,c,ch=img_ex.shape 
        # img_ex=img_ex1[:,:,100]
        #mask_ex = nib.load(mask_path[25]).get_data()
         
        output_img=np.zeros(img_ex.shape)
        patch_ratio = []
        for i in range(16 + 1):
            patch_ratio.append(32 * i)
        for i in range(0,img_ex.shape[2]): 

            patch_ex = slice_to_patch(img_ex[:, :, i], patch_ratio)
            prediction = loaded_model.predict(patch_ex)
            prediction_mask = patch_to_slice(prediction, patch_ratio, input_shape, conf_threshold = 0.97)
            prediction_mask1=img_ex[:,:,0]
            # cv2.imwrite('output.jpg',prediction_mask1)
            
            # self.ids.image.source='output.jpg'
            
            fig, (ax1,ax3) = plt.subplots(1, 2, figsize = ((15, 15)))
          
            ax1.imshow(np.rot90(img_ex[:, :, i], 3), cmap = 'bone')
            ax1.set_title("Image", fontsize = "x-large")
            ax1.grid(False)
      
            ax3.imshow(np.rot90(prediction_mask.reshape((512, 512)), 3), cmap = 'bone')
            ax3.set_title("Mask (Pred)", fontsize = "x-large")
            ax3.grid(False)
            plt.show()
            # plt.close()
            print('Finished')
            
            
                
    def selected(self, filename):
        print ("selected: %s" % filename[0])
        

class MyApp(App):
    def build(self):
        return MyWidget()

if __name__ == '__main__':
    MyApp().run()