import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random

class residual_stack(nn.Module):
    def __init__(self, size, dilation):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            weight_norm(nn.Conv1d(size, size, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(size, size, kernel_size=1))
        )
        self.shortcut = weight_norm(nn.Conv1d(size, size, kernel_size=1))

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


def encoder_sequential(input_size, output_size, *args, **kwargs):
    return nn.Sequential(
        nn.LeakyReLU(0.2),
        weight_norm((nn.ConvTranspose1d(input_size, output_size, *args, **kwargs)))
    )


#-------------melgan генератор-----------------
class GeneratorMel(nn.Module):
    def __init__(self, mel_dim):
        super().__init__()

        factor = [8, 8, 2, 2]

        layers = [
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(mel_dim, 512, kernel_size=7)),
        ]

        input_size = 512
        for f in factor:
            layers += [encoder_sequential(input_size,
                                          input_size // 2,
                                          kernel_size=f * 2,
                                          stride=f,
                                          padding=f // 2 + f % 2)]
            input_size //= 2
            for d in range(3):
                layers += [residual_stack(input_size, 3 ** d)]

        layers += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(32, 1, kernel_size=7)),
            nn.Tanh(),
        ]
        self.generator = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.generator(x)


def decoder_sequential(input_size, output_size, *args, **kwargs):
    return nn.Sequential(
        weight_norm((nn.Conv1d(input_size, output_size, *args, **kwargs))),
        nn.LeakyReLU(0.2, inplace=True)
    )

#-------------Дискриминатор-----------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Каждый Дискриминатор
        self.discriminator = nn.ModuleList([
            # Получаем 16 фич
            nn.Sequential(
                nn.ReflectionPad1d(7),
                weight_norm(nn.Conv1d(1, 16, kernel_size=15)),
                nn.LeakyReLU(0.2, inplace=True) # изменить вход
            ),
            # Четыре понижения размерности в 4 раза
            decoder_sequential(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            decoder_sequential(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            decoder_sequential(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            decoder_sequential(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            # Фичи 1024*32(16,8)
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, padding=2)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Выход 1*32(16,8)
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1))
        ])

    def forward(self, x):
        feature_map = []
        for module in self.discriminator:
            x = module(x)
            feature_map.append(x)
        return feature_map

#-------------3 Дискриминатора-----------------
class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()

        # Каждому Дискриминатору свой блок
        self.block = nn.ModuleList([
            Discriminator() for _ in range(3)
        ])
        # Блок понижающий размерность сигнала    
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
    def forward(self, x):
        result = []
        # Посчитать выходы 3 дискриминаторов
        for idx, module in enumerate(self.block):
            result.append(module(x))
            if idx <= 1:
                # понизить размерность сигнала
                x = self.avgpool(x)
        
        return result


#------------Генератор для wavgan------------------
class GeneratorWav(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv_1 = nn.Sequential(
            nn.ReflectionPad1d(7),
            weight_norm(nn.Conv1d(1, 16, kernel_size=15)),
            nn.LeakyReLU(0.2)
        )
        self.Conv_2 = nn.Sequential(
            weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.LeakyReLU(0.2)
        )
        self.Conv_3 = nn.Sequential(
            weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(0.2)
        )
        self.Conv_4 = nn.Sequential(
            weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
            residual_stack(1024, 3),
            residual_stack(1024, 9),
            nn.LeakyReLU(0.2)
        )
        self.ConvTrans_4 = nn.Sequential(
            weight_norm(nn.ConvTranspose1d(1024, 256, kernel_size=16, stride=4, padding=6)),
            residual_stack(256, 3),
            residual_stack(256, 9),
            nn.LeakyReLU(0.2)
        )
        self.ConvTrans_3 = nn.Sequential(
            weight_norm(nn.ConvTranspose1d(256, 64, kernel_size=16, stride=4, padding=6)),
            residual_stack(64, 3),
            residual_stack(64, 9),
            nn.LeakyReLU(0.2)
        )        
        self.ConvTrans_2 = nn.Sequential(
            weight_norm(nn.ConvTranspose1d(64, 16, kernel_size=16, stride=4, padding=6)),
            residual_stack(16, 3),
            residual_stack(16, 9),
            nn.LeakyReLU(0.2)
        )        
        self.ConvTrans_1 = nn.Sequential(
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(16, 1, kernel_size=7)),
            nn.Tanh()
        )        

        
    def forward(self, x):
        self.x1 = self.Conv_1(x)
        self.x2 = self.Conv_2(self.x1)
        self.x3 = self.Conv_3(self.x2)
        self.x4 = self.Conv_4(self.x3)
        self.x4 = self.ConvTrans_4(self.x4) + self.x3
        self.x3 = self.ConvTrans_3(self.x4) + self.x2
        self.x2 = self.ConvTrans_2(self.x3) + self.x1
        self.x1 = self.ConvTrans_1(self.x2)
        return self.x1

#-------------------------AGAIN-----------------------------
# Нормирующий блок и вычисляющий среднее значение и дисперсию
class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x, mask=None):
        B, C = x.shape[:2]
        
        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
        mn = mn.view(B, C, *((len(x.shape) - 2) * [1]))
        sd = sd.view(B, C, *((len(x.shape) - 2) * [1]))
        
        return mn, sd


    def forward(self, x, return_mean_std=False):
        # Вычисляем среднее значение и дисперсию
        mean, std = self.calc_mean_std(x)
        
        # Нормируем
        x = (x - mean) / std
        
        #  Возвращаем нормированный выход
        if return_mean_std:
            return x, mean, std
        else:
            return x

        

# Сверточный блок в кодировщике
class ConvBlock(nn.Module):
    def __init__(self, c_h):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv1d(c_h, c_h, kernel_size=3, padding=1),
                nn.BatchNorm1d(c_h),
                nn.LeakyReLU()
                )

    def forward(self, x):
        y = self.seq(x)
        return x + y


# Кодировщик
class Encoder(nn.Module):
    def __init__(self, c_in, c_out, n_conv_blocks, c_h):
        super().__init__()
        self.inorm = InstanceNorm()
        self.in_layer = nn.Conv1d(c_in, c_h, kernel_size=1)
        self.conv1d_blocks = nn.ModuleList([ConvBlock(c_h) 
            for _ in range(n_conv_blocks)
        ])
        self.out_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.alpha = 0.1
        
    def forward(self, x):
        # Входная свертка
        y = self.in_layer(x)

        mns = []
        sds = []
        for block in self.conv1d_blocks:
            # Ценртальные свертки
            y = block(y)
            
            # Нормируем и запоминаем среднее и дисперсию
            y, mn, sd = self.inorm(y, return_mean_std=True)
            mns.append(mn)
            sds.append(sd)
        
        # Выходная свертка
        y = self.out_layer(y)
        
        # Активация
        y = 1 / (1+torch.exp(-self.alpha*y))
        
        return (y, mns, sds)

# Декодер
class Decoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, n_conv_blocks):
        super().__init__()
        self.in_layer = nn.Conv1d(c_in, c_h, kernel_size=3, padding=1)
        self.conv_blocks = nn.ModuleList([ConvBlock(c_h) 
            for _ in range(n_conv_blocks)
        ])
        self.rnn = nn.GRU(c_h, c_h, 2)
        self.out_layer = nn.Linear(c_h, c_out)
        
        self.act = nn.LeakyReLU()
        self.inorm = InstanceNorm()


    def forward(self, enc, cond):
        y1, _, _ = enc
        y2, mns, sds = cond
        
        # Нормируем сигнал передаём ему характеристики другого сигнала
        mn, sd = self.inorm.calc_mean_std(y2)
        c = self.inorm(y1)
        c_affine = c * sd + mn
        
        # Входная свертка
        y = self.in_layer(c_affine)
        y = self.act(y)
        
        # Нормируем сигнал и передаём ему характеристики другого сигнала
        for i, (block, mn, sd) in enumerate(zip(self.conv_blocks, mns, sds)):
            y = block(y)
            y = self.inorm(y)
            y = y * sd + mn

        # Пропускаем через RNN
        y = torch.cat((mn, y), dim=2)
        y = y.transpose(1,2)
        y, _ = self.rnn(y)
        y = y[:,1:,:]
        
        # Выходная свертка
        y = self.out_layer(y)
        y = y.transpose(1,2)
        return y



# Модель Again
class AgainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(c_in = 80, c_h = 256,  c_out = 4, n_conv_blocks = 6)
        self.decoder = Decoder(c_in = 4, c_h = 256, c_out = 80, n_conv_blocks = 6)

    def forward(self, source, target = None):  
        if target is None:
            a = random.randint(0, source.shape[2])            
            target = torch.cat((source[:,:,a:], source[:,:,:a]), axis=2)

        y = self.decoder(self.encoder(source), self.encoder(target))
        return y
