import numpy as np
import librosa, os, torch, shutil, random, copy, glob
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils import weight_norm
from hparams import Hparams
from tqdm import tqdm
import soundfile as sf


# Загружаем гиперпараметры
hp = Hparams()


#----------Алгоритм Гриффин Лим-------------
def my_griffin_lim(spectrogram, n_iter=3):
    #копируем спектр
    x_best = copy.deepcopy(spectrogram)
    
    for i in range(n_iter):
        # получаем сигнал
        x_t = librosa.istft(x_best, hp.hop_length, window="hann")
        # получаем спектр
        spec = librosa.stft(x_t, hp.n_fft, hp.hop_length)
        # берём фазу
        phase = np.angle(spec) 
        # Получаем полный спектр исходного сигнала
        x_best =  spectrogram*np.cos(phase) + 1j*spectrogram*np.sin(phase)
    
    # Итоговый сигнал
    x_t = librosa.istft(x_best, hp.hop_length, window="hann")
    return x_t
    
    
#-------Функция обработки датасета--------------
def preprocess(format_wav):
    
    #Создать папку
    train_dir = os.path.join(hp.train_dir)
    os.makedirs(train_dir, exist_ok=True)
    
    # Датасет для тренировки
    dataset = glob.glob(os.path.join(hp.data_dir, '**/*.'+hp.format), recursive=True)
    for train_path in tqdm(dataset):
        # Только название файла
        train_name = os.path.basename(train_path) 
        
        # Обработать звук в мел
        process_audio(train_path, hp.format, train_name, hp.train_dir, it_audio = format_wav)
    
    print('Тренировачный датасет создан')


#----------Обработка аудио в мел, или в Гриффин Лим аудио-------------
def process_audio(wav_path, format = 'wav', wav_name=None, train_dir ='./train', it_audio = True):
    # чтение файла
    wav, sr = librosa.core.load(wav_path, sr=hp.sr)
    
    # Получить спектр
    spectrogram = librosa.stft(y=wav, n_fft=hp.n_fft, hop_length = hp.hop_length, win_length = hp.win_length)
    
    # Получить мелспектрограмму
    output = np.dot(hp.mel_basis, np.abs(spectrogram)).astype(np.float32)
    
    # синтезируем Гриффин Лим аудио
    if it_audio:
        amp = np.maximum(1e-10, np.dot(hp._inv_mel_filtr, output))
        output = my_griffin_lim(amp,5).astype(np.float32)
  
    # Перевести в torch
    output = torch.from_numpy(output)

    # Получаем новый путь до файла
    if wav_name is not None:
        old_wav_path = wav_path
        wav_path = os.path.join(train_dir, wav_name)
        
        # Если было wav то копируем, иначе конвертируем
        if format == 'wav':
            # Копируем wav
            shutil.copyfile(old_wav_path, wav_path) 
        else:
            # Cоздаём wav из другого формата
            sf.write(wav_path.replace('.'+format, '.wav'), wav, hp.sr)    
    
        # Сохраняем Мелспектрограмму, или Гриффин Лим аудио
        if it_audio:
            save_path = wav_path.replace('.'+format, '.glim')
        else:
            save_path = wav_path.replace('.'+format, '.mel')
        torch.save(output, save_path)
        
    return output.unsqueeze(0)
   
   
class MelGanDataset(Dataset):
    def __init__(self, mel_list):
        # Инициализировать список мелспектрограмм
        self.mel_list = mel_list

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        # Считываем mel
        mel = torch.load(self.mel_list[idx])
        
        # Считываем wav
        wav_name = self.mel_list[idx].replace('.mel', '.wav')
        wav, _ = librosa.core.load(wav_name, sr=None)
        wav = torch.from_numpy(wav).float()
        
        # Подгоняем все записи под одну длину, чтобы обучать пакетом
        start = random.randint(0, mel.size(1) - hp.seq_len - 1)
        mel = mel[:, start : start + hp.seq_len]
        start *= hp.hop_length
        wav = wav[start : start + hp.seq_len * hp.hop_length]
        return mel, wav.unsqueeze(0)

class WavGanDataset(Dataset):
    def __init__(self, mel_list, format_wav):
        # Инициализировать список мелспектрограмм
        self.mel_list = mel_list
        self.format_wav = format_wav
        
    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        # Считываем mel, или Гриффин Лим аудио
        input = torch.load(self.mel_list[idx])
        
        # Если это mel синтезировать 
        if not self.format_wav:
            # Амплитуда
            amp = np.maximum(1e-10, np.dot(hp._inv_mel_filtr, input))
            # Синтезируем
            input = my_griffin_lim(amp,5)
            # В тензор
            input = torch.from_numpy(input)
        
        # Считываем wav
        wav_name = self.mel_list[idx].replace('.mel', '.wav').replace('.glim', '.wav')
        wav, _ = librosa.core.load(wav_name, sr=None)
        wav = torch.from_numpy(wav).float()
        
        """
        #Подмешиваем шум
        type_f = np.random.sample()
        if type_f<0.3:
            # К Гриффин лим сигналу подмешиваем шум
            input = input + torch.from_numpy(np.random.sample(input.shape) * np.random.sample()* 0.01).float()
        elif type_f<0.6:
            # В качестве исходных данный конечный сигнал с шумом
            input  = wav + torch.from_numpy(np.random.sample(wav.shape) * np.random.sample()* 0.01).float()
        """

        # Подгоняем все записи под одну длину, чтобы обучать пакетом
        start = random.randint(0,input.size(-1) - hp.seq_len * hp.hop_length - 1)
        input = input[start : start + hp.seq_len * hp.hop_length]
        wav = wav[start : start + hp.seq_len * hp.hop_length]
        return input.unsqueeze(0), wav.unsqueeze(0)

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


