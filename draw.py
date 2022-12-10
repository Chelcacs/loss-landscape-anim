# import sys
# sys.path.append("loss_landscape_anim_or\\loss_landscape_anim")
from loss_landscape_anim.main import loss_landscape_anim, LeNet, VGG
# from datamodule import CIFAR10DataModule
from loss_landscape_anim.datamodule import MNISTDataModule

bs = 16 
lr = 1e-3 
datamodule = MNISTDataModule(batch_size=bs, n_examples=3000) 
model = LeNet(learning_rate=lr) 
model2 = VGG("VGG11",lr)

optim_path, loss_steps, accu_steps = loss_landscape_anim( 
    n_epochs=10, 
    model=model2, 
    datamodule=datamodule,
    optimizer= "sgd", 
    model_filename='vgg11.pt',
    output_filename='vgg11.gif',
    giffps=15, 
    seed=180102, 
    load_model=False, 
    output_to_file=True, 
    return_data=True, # 如果需要的话可选的返回值
    gpus=1 # 如果可用的话启用 GPU 训练
)