import argparse, os, glob, time, librosa, torch
from tqdm import tqdm
from utils import  preprocess, process_audio, MelGanDataset
from utils import GeneratorMel, MultiScale
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
    vocoder = GeneratorMel(hp.n_mels)
    
    # Загружаем веса
    ckpt = torch.load(hp.mel_checkpoint, map_location='cpu')
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
        
        # Загружаем Мелспектр, или получаем его из звука
        if format_wav:
            # Получаем Мелспектр из аудио
            mel = process_audio(test_path, it_audio = False)
        else:
            # Загружаем Мелспектр
            mel = torch.load(test_path).unsqueeze(0)  
        
        # Сгенерировать сигнал из mel 
        audio_output = vocoder(mel)
        
        # Убрать лишнюю размерность, градиенты и перевести в numpy
        audio_output = audio_output.squeeze().detach().numpy()
        
        # Сохранить в файл
        sf.write(generate_dir +'/mel_' + test_name, audio_output, hp.sr)


#-------Функция обучении модели--------------
def train():
    # Создать папку для логов
    save_dir = os.path.join(hp.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Загрузить список тренировачных mel
    mel_list = glob.glob(os.path.join(hp.train_dir, '*.mel'))
    
    # Создать Датасет
    trainset = MelGanDataset(mel_list)
    train_loader = DataLoader(trainset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

    # Загрузить тестовый датасет
    test_mels = glob.glob(os.path.join(hp.test_dir, '*.wav'))
    testset = [process_audio(test_mel, it_audio = False) for test_mel in test_mels]
    
    # создать Generator и 3*Discriminator
    G = GeneratorMel(hp.n_mels).cuda()
    D = MultiScale().cuda()
    g_optimizer = optim.Adam(G.parameters(), lr=hp.lr, betas=(hp.betas1, hp.betas2))
    d_optimizer = optim.Adam(D.parameters(), lr=hp.lr, betas=(hp.betas1, hp.betas2))

    # Загружаем модель
    step, epochs = 0, 0
    if hp.mel_checkpoint is not None:
        print("Загрузка чекпоинтов")
        ckpt = torch.load(hp.mel_checkpoint)
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
        for (mel, audio) in train_loader:
            # Помещаем входной и выходной  файл в видеокарту
            mel = mel.cuda()
            audio = audio.cuda()

            # Получаем сигнал из спектра 16*1*8192
            fake_audio = G(mel)
            # Получаем отклик на созданное аудио без градиентов 3*7*16*16*8192(4096,2048)
            d_fake_detach = D(fake_audio.cuda().detach())
            # Получаем отклик на созданное аудио
            d_fake = D(fake_audio.cuda())
            # Получаем отклик на реальное аудио
            d_real = D(audio)    

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
            
            # вычисляем градиенты и делаем шаг 
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
                }, save_dir +'/mel_ckpt_%dk.pt' % (step // 10000))
                
                if testset:
                    print("Синтезируем тестовые файлы")
                    with torch.no_grad():
                        for i, mel_test in enumerate(testset):
                            audio_output = G(mel_test.cuda())
                            audio_output = audio_output.squeeze().cpu().numpy()
                            sf.write(save_dir +'/mel_gen_%d_%dk_%d.wav' % (epoch, step // 1000, i), audio_output, hp.sr)
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
        train()
    elif args.run == 'preprocess' or args.run == 'p':
        preprocess(False)
    elif args.run == 'generate' or args.run == 'g':
        generate(format_wav)        
    else:
        print("Введите корректную функцию, которую надо запустить (preprocess, train, generate)")  
    