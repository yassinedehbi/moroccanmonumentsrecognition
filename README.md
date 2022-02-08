preparedata.py : for data augmentation by splitting, rotating, shifting, resizing images.

split_data.py: split data to train, validation and test (0.8,0.1,0.1). The data is stored in finaldata dir 

main.py : load and prepare data. Build and fit  the model which is saved in model dir.

evaluation.py and evaluation_metrics.py for model evaluation.

predict.py `<image_path>` : make prediction and return the corresponding zone


The final model is modell21.h5
