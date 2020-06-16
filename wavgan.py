import argparse, os, glob, time, sys, librosa, torch
from tqdm import tqdm
from utils import  preprocess, process_audio, WavGanDataset, my_griffin_lim
from utils import GeneratorWav, MultiScale
import numpy as np
import soundfile as sf

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hparams import Hparams



#-------Функция генерации звука--------------
def generate(format_wav):
    # Создаём папку с выходными файлами
    generate_dir = os.path.join(hp.generate_dir, 'output')    
    os.makedirs(generate_dir, exist_ok=True)
    
    # Создаём вокодер
    vocoder = GeneratorWav()
    
    # Загружаем веса
    ckpt = torch.load(hp.wav_checkpoint, map_location='cpu')
    vocoder.load_state_dict(ckpt['G'])
    
    # Информация о чекпоинте
    step = ckpt['step']
    epochs = int(ckpt['epoch'])
    print('Чекпоинт загружен: Эпоха %d, Шаг %d' % (epochs, step))
    
    # Загружаем тестовые файлы
    if format_wav:
        testset = glob.glob(os.path.join(hp.generate_dir, '*.wav'))
    else:
        testset = glob.glob(os.path.join(hp.generate_dir, '*.mel'))
    
    for i, test_path in enumerate(tqdm(testset)):
        test_name = os.path.basename(test_path).replace('.mel', '.wav')
        
        # Загружаем Гриффин Лим звук, или получаем его из Мел
        if format_wav:
            # Читаем wav
            audio_input, sr = librosa.core.load(test_path, sr=hp.sr)
        else:
            # Читаем mel спектр
            mel = torch.load(test_path)
            
            # Получаем амплитудный спектр
            amp = np.maximum(1e-10, np.dot(hp._inv_mel_filtr, mel))
            
            # Синтезируем использую Гриффин Лим
            audio_input = my_griffin_lim(amp,5)
            
        # Выкидываем хвост, чтобы длина делилась на 64
        if (len(audio_input)%64):
            audio_input = audio_input[:-(len(audio_input)%64)]
        
        # Перевести в торч и добавить размерности
        audio_input = torch.from_numpy(audio_input).unsqueeze(0).unsqueeze(0)   

        # Сгенерировать из mel сигнал
        audio_output = vocoder(audio_input)
        
        # Убрать лишнюю размерность, градиенты и перевести в numpy
        audio_output = audio_output.squeeze().detach().numpy()
        
        # Сохранить в файл
        sf.write(generate_dir +'/wav_' + test_name, audio_output, hp.sr)
        
#-------Функция обучении модели--------------
def train(format_wav):
    # Создать папку для логов
    save_dir = os.path.join(hp.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Загрузить список тренировачных mel
    if format_wav:
        # Синтезированные Гриффин Лим файлы
        mel_list = glob.glob(os.path.join(hp.train_dir, '*.glim'))
    else:
        # Мел файлы
        mel_list = glob.glob(os.path.join(hp.train_dir, '*.mel'))
    
    # Создать Датасет
    trainset = WavGanDataset(mel_list,format_wav)
    train_loader = DataLoader(trainset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

    # Загрузить тестовый датасет
    test_wavs = glob.glob(os.path.join(hp.test_dir, '*.wav'))
    testset = [process_audio(test_mel, it_audio = True).unsqueeze(0) for test_mel in test_wavs]

    # создать Generator и 3*Discriminator
    G = GeneratorWav().cuda()
    D = MultiScale().cuda()
    g_optimizer = optim.Adam(G.parameters(), lr=hp.lr, betas=(hp.betas1, hp.betas2))
    d_optimizer = optim.Adam(D.parameters(), lr=hp.lr, betas=(hp.betas1, hp.betas2))

    # Загружаем модель
    step, epochs = 0, 0
    if hp.wav_checkpoint is not None:
        print("Загрузка чекпоинтов")
        ckpt = torch.load(hp.wav_checkpoint)
        G.load_state_dict(ckpt['G'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        D.load_state_dict(ckpt['D'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        step = ckpt['step']
        epochs = int(ckpt['epoch'])
        print('Чекпоинт загружен: Эпоха %d, Шаг %d' % (epochs, step))

    #Попытка оптимизировать сеть
    torch.backends.cudnn.benchmark = True

    # Процесс обучения
    start = time.time()
    for epoch in range(epochs, hp.max_epoch):
        for (audio_input, audio_output) in train_loader:
            # Помещаем входной и выходной  файл в видеокарту
            audio_input = audio_input.cuda()
            audio_output = audio_output.cuda()

            # Получаем выходной сигнал из входного 16*1*8192
            fake_audio = G(audio_input)
            # Получаем отклик на созданное аудио без градиентов 3*7*16*16*8192(4096,2048)
            d_fake_detach = D(fake_audio.cuda().detach())
            # Получаем отклик на созданное аудио
            d_fake = D(fake_audio.cuda())
            # Получаем отклик на реальное аудио
            d_real = D(audio_output)    

            # ------------Дискриминатор---------------
            # Считаем ошибку на отклике на реальный сигнал. чем больше d_real тем лучше
            d_loss_real = 0
            for scale in d_real:
                d_loss_real += F.relu(1 - scale[-1]).mean()

            # Считаем ошибку на отклике на созданном сигнале без градиента. чем меньше d_loss_fake, тем лучше
            d_loss_fake = 0
            for scale in d_fake_detach:
                d_loss_fake += F.relu(1 + scale[-1]).mean()
            
            # Суммарная ошибка
            d_loss = d_loss_real + d_loss_fake
            
            # Вычисляем градиенты и делаем шаг 
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ---------------Генератор----------------------
            # Считаем ошибку на отклик созданного сигнала с градиентом
            g_loss = 0
            for scale in d_fake:
                g_loss += -scale[-1].mean()

            # Считаем ошибку между откликом на реальный сигнал и на созданный
            feature_loss = 0
            for i in range(len(d_fake)):
                for j in range(len(d_fake[i]) - 1):
                    feature_loss += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

            # Суммарная ошибка
            g_loss += hp.lambda_feat * feature_loss
            
            # вычисляем градиенты и делаем шаг 
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # выводим ошибку
            step += 1
            if step % hp.log_interval == 0:
                print('Эпоха: %-5d, Шаг: %-7d, D_loss: %.05f, G_loss: %.05f, ms/batch: %5.2f' %
                    (epoch, step, d_loss, g_loss, 1000 * (time.time() - start) / hp.log_interval))
                start = time.time()
            
            # Сохраняем модель и синтезируем тестовые файлы
            if step % hp.save_interval == 0:
                print("Сохраняем модель")
                torch.save({
                    'G': G.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'D': D.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'step': step,
                    'epoch': epoch,
                }, save_dir +'/wav_ckpt_%dk.pt' % (step // 10000))
                
                if testset:
                    print("Синтезируем тестовые файлы")
                    with torch.no_grad():
                        for i, audio_input in enumerate(testset):
                            audio_output = G(audio_input.cuda())
                            audio_output = audio_output.squeeze().detach().cpu().numpy()
                            sf.write(save_dir +'/wav_gen_%d_%dk_%d.wav' % (epoch, step // 1000, i), audio_output, hp.sr)
                else:
                    print("Нет файлов для тестирования. Поместите их в test_dir")

if __name__ == "__main__":
    # Загружаем гиперпараметры
    hp = Hparams()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", default='train', help=\
        "Введите функцию, которую надо запустить (preprocess, train, generate)")
    parser.add_argument("-f", "--format_wav", default='wav', help="Работать с wav, или mel")    
    args = parser.parse_args()
    
    if args.format_wav == 'wav':
        format_wav = True
    elif args.format_wav == 'mel':
        format_wav = False   
    else:
        print("Введите корректный формат данных (wav, mel)")
        sys.exit(0)
        
    if args.run == 'train' or args.run == 't':
        train(format_wav)
    elif args.run == 'preprocess' or args.run == 'p':
        preprocess(format_wav)
    elif args.run == 'generate' or args.run == 'g':
        generate(format_wav)        
    else:
        print("Введите корректную функцию, которую надо запустить (preprocess, train, generate)")  
    