import os,argparse,torch,librosa,glob
import matplotlib.pyplot as plt
from utils import AgainDataset,preprocess,process_audio, my_griffin_lim
from models import AgainModel,GeneratorMel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from hparams import Hparams
import librosa.display
import soundfile as sf

# Гиперпараметры
hp = Hparams()

# Устройство, видеокарта или процессор
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Тренировка
def train(): 
    # Датасеты
    train_set = AgainDataset(dset='train')
    dev_set = AgainDataset(dset='eval')
    
    # Даталоадер
    train_loader = DataLoader(train_set, batch_size = hp.batch_size, shuffle = True)
    dev_loader = DataLoader(dev_set, batch_size = hp.batch_size*3)
    
    # Модель
    model = AgainModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0005,betas = [0.9,0.999],amsgrad = True,weight_decay = 0.0001 )
    criterion_l1 = torch.nn.L1Loss()
    steps = 0

    # Загрузить чекпоинт
    if hp.again_checkpoint != '':
        ckpt = torch.load(hp.again_checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        steps = ckpt['steps']
        print('Чекпоинт загружен: Шаг %d' % (steps))
    
    
    # Тренировка
    while steps <= hp.again_steps:
        train_bar = tqdm(train_loader)
        model.train()
        for data in train_bar:
            steps += 1
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Пропускаем через модель 
            x = data.to(device)
            y = model(x)
            
            # Считаем ошибку
            loss = criterion_l1(y, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),  max_norm=10)
            optimizer.step()
            
            # Выводим статистику
            if steps % hp.log_interval == 0:
                train_bar.set_postfix({'loss_rec': loss.item(),'steps':steps})
            
            # Сохраняем модель
            if steps % hp.save_interval == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                }, os.path.join(hp.save_dir, f'again_{steps//1000}k.pth'))                 
        
        # Валидация
        model.eval()
        dev_bar = tqdm(dev_loader)
        for data in dev_bar:
            with torch.no_grad():
                x = data.to(device)
                y = model(x)
                loss = criterion_l1(y, x)
                dev_bar.set_postfix({'loss_rec': loss.item()})

# Генерация     
def generate(args):  
    # Создаём папку с выходными файлами
    generate_dir = os.path.join(hp.generate_dir, 'output')    
    os.makedirs(generate_dir, exist_ok=True)
    
    # Создать модель
    model = AgainModel().to(device)
    model.eval()
    
    # Загрузить чекпоинт
    ckpt = torch.load(hp.again_checkpoint)
    model.load_state_dict(ckpt['model'])
    
    # Создаём вокодер и Загружаем веса
    vocoder = GeneratorMel(hp.n_mels).to(device) 
    ckpt = torch.load(hp.mel_checkpoint, map_location=device)
    vocoder.load_state_dict(ckpt['G'])
    
    # Загрузить звук
    source = process_audio(args.source, 'mel', wav_name=None, it_audio = False)[0]
    target = process_audio(args.target, 'mel', wav_name=None, it_audio = False)[0]
    
    # Обрезать до одной длины
    if hp.seglen is not None:
        target = target[:,:hp.seglen]
        source = source[:,:hp.seglen]
       
    # Получить новый путь   
    source_path = os.path.join(generate_dir, f'{os.path.basename(args.source).replace(".wav","")}')
    target_path = os.path.basename(args.target).replace('.wav','')
    
    # Получить новый файл
    with torch.no_grad():    
        mel_output = model(source.unsqueeze(0).to(device), target.unsqueeze(0).to(device)) 
        audio_output = vocoder(10**mel_output).squeeze().cpu().numpy()
        #amp = np.maximum(1e-10, np.dot(hp._inv_mel_filtr,10**(mel_output.squeeze().cpu().numpy()) ))
        #audio_output = my_griffin_lim(amp,5)
        sf.write(source_path+f'_to_{target_path}'+'.wav', audio_output, hp.sr)
        
        
    # Нарисовать Мелспектрограммы
    librosa.display.specshow(source.numpy(), cmap='viridis')
    plt.savefig(source_path+'.png')
    librosa.display.specshow(mel_output.squeeze().cpu().numpy(), cmap='viridis')
    plt.savefig(source_path+f'_to_{target_path}'+'.png')

 
if __name__ == '__main__':
    # Парсим запрос
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", default='train', help=\
        "Введите функцию, которую надо запустить (preprocess, train, generate)")
    parser.add_argument('--source', '-s', help='Source path. A .wav file or a directory containing .wav files.')
    parser.add_argument('--target', '-t', help='Target path. A .wav file or a directory containing .wav files.')        
    args = parser.parse_args()        
    
    # Выбираем режим работы
    if args.run == 'train' or args.run == 't':
        train()
    elif args.run == 'preprocess' or args.run == 'p':
        preprocess(False)
    elif args.run == 'generate' or args.run == 'g':
        generate(args)        
    else:
        print("Введите корректную функцию, которую надо запустить (preprocess, train, generate)") 

    

