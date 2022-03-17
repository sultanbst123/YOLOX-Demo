
import numpy as np
from keras_cv_attention_models.yolox import * # import all yolox model
from keras_cv_attention_models.coco import data
import matplotlib.pyplot as plt
import gradio as gr
 
# semua yolox model
choices = ["YOLOXNano", "YOLOXTiny", "YOLOXS", "YOLOXM", "YOLOXL", "YOLOXX"]

def main(input_img, models):
    
    #
    fig, ax = plt.subplots() # pakai ini,jika tidak akan muncul error

    # YOLOXNano models
    if models == "YOLOXNano": 
        model = YOLOXNano(pretrained="coco") 

    # YOLOXTiny models
    elif models == "YOLOXTiny": 
        model = YOLOXTiny(pretrained="coco") 

    # YOLOXS models   
    elif models == "YOLOXS":       
        model = YOLOXS(pretrained="coco") 

    # YOLOXM models    
    elif models == "YOLOXM": 
        model = YOLOXM(pretrained="coco") 

    # YOLOXL models    
    elif models == "YOLOXL":
        model = YOLOXL(pretrained="coco") 

    # YOLOXX models   
    elif models == "YOLOXX":      
        model = YOLOXX(pretrained="coco") 

    # pass    
    else:  
        pass

    # image pre processing yolox
    preds = model(model.preprocess_input(input_img))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]
    data.show_image_with_bboxes(input_img, bboxs, lables, confidences, num_classes=100,label_font_size=17, ax=ax)

    return fig

# define params

input = [gr.inputs.Image(shape=(2000, 1500),label = "Input Image"),
         gr.inputs.Dropdown(choices= choices, type="value", default='YOLOXS', label="Model")]

output = gr.outputs.Image(type="plot", label="Output Image")

title = "YOLOX Demo"

example = [["images_1.jpeg ","YOLOXM"],["images_2.jpeg","YOLOXS"],["images.jpeg","YOLOXL"]]

description = "Demo for YOLOX(Object Detection). Models are YOLOXNano - YOLOXX"           

article = "<a href='https://github.com/Megvii-BaseDetection/YOLOX' target='_blank'><u>YOLOX </u></a>is an anchor-free version of YOLO, with a simpler design but better performance!<br><br><p style='text-align: center'>Untuk penjelasan lihat di <a href='https://github.com/sultanbst123/Text_summarization-id2id' target='_blank'><u>repo ku </u>😁</a></p>"

# deploy
iface = gr.Interface(main, 
                    inputs = input,
                    outputs = output, 
                    title = title,
                    article = article,
                    description = description, 
                    examples = example, 
                    theme = "dark") 
                    
iface.launch(debug = True)