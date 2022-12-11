from loss_landscape_anim.main import loss_landscape_anim, LeNet, VGG
from loss_landscape_anim.model import CNN
from loss_landscape_anim.datamodule import MNISTDataModule


bs = 16 
lr = 1e-3 
datamodule = MNISTDataModule(batch_size=bs, n_examples=3000) 
# model = LeNet(learning_rate=lr) 
model = CNN(learning_rate=lr)

optim_path, loss_steps, accu_steps = loss_landscape_anim( 
    n_epochs=10, 
    model=model, 
    datamodule=datamodule,
    optimizer= "sgd", 
    model_filename='alexnet.pt',
    output_filename='alexnet.gif',
    giffps=15, 
    seed=180102, 
    load_model=False, 
    output_to_file=True, 
    return_data=True, # 如果需要的话可选的返回值
    gpus=1 # 如果可用的话启用 GPU 训练
)