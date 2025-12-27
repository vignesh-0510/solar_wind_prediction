from  torchinfo import summary

def save_summary(file_path, model, input_shape):
    with open(file_path,'w+') as f:
        f.writelines(str(summary(model, input_size=input_shape)))