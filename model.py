import time
import torch
import datetime
import numpy as np
import torch.nn as nn

from function import embedding_lookup, conv2d, deconv2d, fc, TrainDataProvider

curr_epoch = 0
# epoch 확인
def checkEpoch():
    global curr_epoch
    if curr_epoch == 300:
        curr_epoch = 0

    return curr_epoch

# 생성자(Generator)
def Generator(images, En, De, embeddings, embedding_ids, GPU=False, encode_layer=False):
    # 인코드 결과
    encoded_result, encode_layers = En(images)

    local_embeddings = embedding_lookup(embeddings, embedding_ids, GPU=GPU)
    if GPU:
        encoded_result = encoded_result.cuda()
        local_embeddings = local_embeddings.cuda()
    # 인코드 결과와 embedding 결합
    embedded = torch.cat((encoded_result, local_embeddings), 1)
    # 가짜 이미지 생성
    fake_target = De(embedded, encode_layers)
    if encode_layer:
        return fake_target, encoded_result, encode_layers
    else:
        return fake_target, encoded_result

# 인코더(Encoder)
class Encoder(nn.Module):
    def __init__(self, c_in=1):
        super(Encoder, self).__init__()
        self.conv1 = conv2d(c_in, 64, k_size=5, stride=2, pad=2, lrelu=False, bn=False) # 64x64x64
        self.conv2 = conv2d(64, 128, k_size=5, stride=2, pad=2) # 128x32x32
        self.conv3 = conv2d(128, 256, k_size=4, stride=2, pad=1) # 256x16x16
        self.conv4 = conv2d(256, 512) # 512x8x8
        self.conv5 = conv2d(512, 512) # 512x4x4
        self.conv6 = conv2d(512, 512) # 512x2x2
        self.conv7 = conv2d(512, 512) # 512x1x1
        self.conv8 = conv2d(512, 512) # 512x1x1

    def forward(self, images):
        encode_layers = dict()

        e1 = self.conv1(images)
        encode_layers['e1'] = e1
        e2 = self.conv2(e1)
        encode_layers['e2'] = e2
        e3 = self.conv3(e2)
        encode_layers['e3'] = e3
        e4 = self.conv4(e3)
        encode_layers['e4'] = e4
        e5 = self.conv5(e4)
        encode_layers['e5'] = e5
        e6 = self.conv6(e5)
        encode_layers['e6'] = e6
        e7 = self.conv7(e6)
        encode_layers['e7'] = e7
        encoded_result = self.conv8(e7)
        encode_layers['e8'] = encoded_result

        return encoded_result, encode_layers

# 디코더(Decoder)
class Decoder(nn.Module):
    def __init__(self, img_dim=1, embedded_dim=640):
        super(Decoder, self).__init__()
        self.deconv1 = deconv2d(embedded_dim, 512, dropout=True)    #1024x1x1
        self.deconv2 = deconv2d(1024, 512, dropout=True, k_size=4) #1024x2x2
        self.deconv3 = deconv2d(1024, 512, k_size=5, dropout=True) #1024x4x4
        self.deconv4 = deconv2d(1024, 512, k_size=4, stride=2)  # 1024x8x8
        self.deconv5 = deconv2d(1024, 256, k_size=4, stride=2)  # 512x16x16
        self.deconv6 = deconv2d(512, 128, k_size=4, stride=2)   # 256x32x32
        self.deconv7 = deconv2d(256, 64, k_size=4, stride=2)   # 128x64x64
        self.deconv8 = deconv2d(128, img_dim, k_size=4, stride=2, bn=False) # 1x128x128

    def forward(self, embedded, encode_layers):
        d1 = self.deconv1(embedded)
        d1 = torch.cat((d1, encode_layers['e7']), dim=1)
        d2 = self.deconv2(d1)
        d2 = torch.cat((d2, encode_layers['e6']), dim=1)
        d3 = self.deconv3(d2)
        d3 = torch.cat((d3, encode_layers['e5']), dim=1)
        d4 = self.deconv4(d3)
        d4 = torch.cat((d4, encode_layers['e4']), dim=1)
        d5 = self.deconv5(d4)
        d5 = torch.cat((d5, encode_layers['e3']), dim=1)
        d6 = self.deconv6(d5)
        d6 = torch.cat((d6, encode_layers['e2']), dim=1)
        d7 = self.deconv7(d6)
        d7 = torch.cat((d7, encode_layers['e1']), dim=1)
        d8 = self.deconv8(d7)
        fake_target = torch.tanh(d8)

        return fake_target

# 판별자(Discriminator)
class Discriminator(nn.Module):
    def __init__(self, category_num, c_in=2):
        super(Discriminator, self).__init__()
        self.conv1 = conv2d(c_in, 64, bn=False)  # 64x64x64
        self.conv2 = conv2d(64, 128)    # 128x32x32
        self.conv3 = conv2d(128, 256)   # 256x16x16
        self.conv4 = conv2d(256, 512)   # 512x8x8
        self.fc1 = fc(512 * 8 * 8, 1)
        self.fc2 = fc(512 * 8 * 8, category_num)

    def forward(self, images):
        batch_size = images.shape[0]
        disc1 = self.conv1(images)
        disc2 = self.conv2(disc1)
        disc3 = self.conv3(disc2)
        disc4 = self.conv4(disc3)

        disc4 = disc4.reshape(batch_size, -1)

        tf_loss_logit = self.fc1(disc4)
        tf_loss = torch.sigmoid(tf_loss_logit)  # 이미지 t/f loss
        cat_loss = self.fc2(disc4)     # category loss

        return tf_loss, tf_loss_logit, cat_loss

# 학습
class Trainer:
    def __init__(self, GPU, data_dir, embed_dir, fonts_num, batch_size, img_size):
        self.GPU = GPU
        self.data_dir = data_dir
        self.embed_dir = embed_dir
        self.fonts_num = fonts_num
        self.batch_size = batch_size
        self.img_size = img_size

        self.embeddings = torch.load(embed_dir + 'EMBEDDINGS.pkl')
        self.embedding_num = self.embeddings.shape[0]
        self.embedding_dim = self.embeddings.shape[3]

        self.data_provider = TrainDataProvider(self.data_dir)
        self.total_batches = self.data_provider.get_total_batch_num(self.batch_size)
        print("total batches:", self.total_batches)

    def train(self, max_epoch, schedule, save_model_path, lr=0.001,
              log_step=100, fine_tune=False, restore=None, from_model_path=False):

        global curr_epoch

        # penalty
        if not fine_tune:
            L1_penalty, Lconst_penalty = 100, 15
        else:
            L1_penalty, Lconst_penalty = 500, 1000

        En = Encoder()
        De = Decoder()
        D = Discriminator(category_num=self.fonts_num)

        # gpu 할당
        if self.GPU:
            En.cuda()
            De.cuda()
            D.cuda()

        # 학슴 모델 불러오기
        if restore:
            encoder_name, decoder_name, discriminator_name = restore
            prev_epoch = int(encoder_name.split('-')[0])
            En.load_state_dict(torch.load(from_model_path + encoder_name))
            De.load_state_dict(torch.load(from_model_path + decoder_name))
            D.load_state_dict(torch.load(from_model_path + discriminator_name))
            print("model is restored")
        else:
            prev_epoch = 0

        l1_criterion = nn.L1Loss(size_average=True).cuda()      # MAE(=L1)
        bce_criterion = nn.BCEWithLogitsLoss(size_average=True).cuda()
        mse_criterion = nn.MSELoss(size_average=True).cuda()

        G_parameters = list(En.parameters()) + list(De.parameters())

        g_optim = torch.optim.Adam(G_parameters, betas=(0.5, 0.999))
        d_optim = torch.optim.Adam(D.parameters(), betas=(0.5, 0.999))

        for epoch in range(max_epoch):
            if (epoch + 1) % schedule == 0:
                updated_lr = max(lr / 2, 0.0002)
                for params in d_optim.param_groups:
                    params['lr'] = updated_lr
                for params in g_optim.param_groups:
                    params['lr'] = updated_lr

                lr = updated_lr

            train_batch_iter = self.data_provider.get_iter(self.batch_size)
            for i, batch in enumerate(train_batch_iter):
                font_ids, _, batch_images = batch
                embedding_ids = font_ids

                if self.GPU:
                    batch_images = batch_images.cuda()

                real_target = batch_images[:, 0, :, :]
                real_target = real_target.view([self.batch_size, 1, self.img_size, self.img_size])
                real_source = batch_images[:, 1, :, :]
                real_source = real_source.view([self.batch_size, 1, self.img_size, self.img_size])

                fake_target, encoded_result = Generator(real_source, En, De, self.embeddings,
                                                        embedding_ids, GPU=self.GPU)

                real_TS = torch.cat([real_source, real_target], dim=1)
                fake_TS = torch.cat([real_source, fake_target], dim=1)

                real_score, real_score_logit, real_cat_logit = D(real_TS)
                fake_score, fake_score_logit, fake_cat_logit = D(fake_TS)

                # 인코드 값 loss
                encoded_fake = En(fake_target)[0]
                const_loss = Lconst_penalty * mse_criterion(encoded_result, encoded_fake)

                real_category = torch.from_numpy(np.eye(self.fonts_num)[embedding_ids]).float()
                if self.GPU:
                    real_category = real_category.cuda()
                # 카테고리 loss
                real_category_loss = bce_criterion(real_cat_logit, real_category)
                fake_category_loss = bce_criterion(fake_cat_logit.clone().detach(), real_category.clone().detach())
                category_loss = (real_category_loss + fake_category_loss)*0.5

                if self.GPU:
                    one_labels = torch.ones([self.batch_size, 1]).cuda()
                    zero_labels = torch.zeros([self.batch_size, 1]).cuda()
                else:
                    one_labels = torch.ones([self.batch_size, 1])
                    zero_labels = torch.zeros([self.batch_size, 1])

                # 이미지 Discriminate
                real_binary_loss = bce_criterion(real_score_logit, one_labels)
                fake_binary_loss = bce_criterion(fake_score_logit, zero_labels)
                binary_loss = real_binary_loss + fake_binary_loss

                # 이미지 loss
                l1_loss = L1_penalty * l1_criterion(real_target, fake_target)
                cheat_loss = bce_criterion(fake_score_logit.clone().detach(), one_labels)

                # g_loss, d_loss
                # g_loss = cheat_loss.clone().detach() + l1_loss.clone().detach() + fake_category_loss.clone().detach() + const_loss.clone().detach()
                g_loss = l1_loss + const_loss + cheat_loss + fake_category_loss
                d_loss = binary_loss + category_loss

                # 판별자/생성자 학습
                D.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optim.step()

                En.zero_grad()
                De.zero_grad()
                g_loss.backward(retain_graph=True)
                g_optim.step()

                # logging
                if (i + 1) % log_step == 0:
                    time_ = time.time()
                    time_stamp = datetime.datetime.fromtimestamp(time_).strftime('%H:%M:%S')
                    log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f' % \
                                 (int(prev_epoch) + epoch + 1, int(prev_epoch) + max_epoch,
                                  i + 1, self.total_batches, l1_loss.item(), d_loss.item(), g_loss.item())
                    print(time_stamp, log_format)

            curr_epoch += 1

        # save model
        torch.save(En.state_dict(), save_model_path + 'Encoder.pkl')
        torch.save(De.state_dict(), save_model_path + 'Decoder.pkl')
        torch.save(D.state_dict(), save_model_path + 'Discriminator.pkl')
