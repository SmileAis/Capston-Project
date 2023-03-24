import torch

from model import Trainer
from function import init_embedding


# pretrain1
def pretrain1():
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dir=  'model/pretrain1/'
    embeddings = init_embedding(40, 128)
    torch.save(embeddings, embed_dir + 'EMBEDDINGS.pkl')

    data_dir = 'data/pretrain/'

    fonts_num = 31
    batch_size = 20
    img_size = 128

    max_epoch = 30
    schedule = 10
    save_model_path = 'model/pretrain1/'

    myTrainer = Trainer(GPU, data_dir, embed_dir, fonts_num, batch_size, img_size)
    myTrainer.train(max_epoch=max_epoch, schedule=schedule, save_model_path=save_model_path, lr=0.001,
                    log_step=100, fine_tune=False, restore=None, from_model_path=False)

# pretrain2
def pretrain2():
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/pretrain/'
    embed_dir = 'model/pretrain1/'

    fonts_num = 31
    batch_size = 20
    img_size = 128

    max_epoch = 150
    schedule = 20
    save_model_path = 'model/pretrain2/'
    from_model_path = 'model/pretrain1/'
    restore = ['30-0921-14:35-Encoder.pkl', '30-0921-14:35-Decoder.pkl', '30-0921-14:35-Discriminator.pkl']

    myTrainer = Trainer(GPU, data_dir, embed_dir, fonts_num, batch_size, img_size)
    myTrainer.train(max_epoch=max_epoch, schedule=schedule,save_model_path=save_model_path, lr=0.001,
                    log_step=100, fine_tune=True, restore=restore, from_model_path=from_model_path)
