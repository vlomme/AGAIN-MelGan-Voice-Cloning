import numpy as np
import librosa, os, torch, shutil, random, copy, glob
from torch.utils.data import Dataset
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
    eval_dir = os.path.join(hp.eval_dir)
    os.makedirs(eval_dir, exist_ok=True)    
    # Датасет для тренировки
    dataset = glob.glob(os.path.join(hp.data_dir, '**/*.'+hp.format), recursive=True)
    for train_path in tqdm(dataset):
        # Только название файла
        train_name = os.path.basename(train_path) 
        
        # Обработать звук в мел
        process_audio(train_path, hp.format, train_name,  it_audio = format_wav)
    
    print('Тренировачный датасет создан')


#----------Обработка аудио в мел, или в Гриффин Лим аудио-------------
def process_audio(wav_path, format = 'wav', wav_name=None,  it_audio = True):
    # чтение файла
    wav, sr = librosa.core.load(wav_path, sr=hp.sr)
    
    # Удалить тишину
    wav, _ = librosa.effects.trim(wav, top_db=hp.trim)

    # Получить спектр
    spectrogram = librosa.stft(y=wav, n_fft=hp.n_fft, hop_length = hp.hop_length, win_length = hp.win_length)
    
    # Получить мелспектрограмму
    output = np.dot(hp.mel_basis, np.sqrt(np.abs(spectrogram**2)))
    output = np.log10(output).astype(np.float32)

    
    # синтезируем Гриффин Лим аудио
    if it_audio:
        amp = np.maximum(1e-10, np.dot(hp._inv_mel_filtr, 10**output))
        output = my_griffin_lim(amp,5).astype(np.float32)
  
    # Перевести в torch
    output = torch.from_numpy(output)

    # Получаем новый путь до файла
    if wav_name is not None:
        #print(wav_name)
        if wav_name[3] =='8':
            dir_folder = hp.eval_dir
        else:
            dir_folder = hp.train_dir
        old_wav_path = wav_path
        #print(dir_folder, wav_name)
        wav_path = os.path.join(dir_folder, wav_name)
        
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
   
# Датасет 
class AgainDataset(Dataset):
    def __init__(self, dset):
        super().__init__()
        print(f'Загружаем {dset} датасет в память')
        dataset = tqdm(glob.glob(os.path.join(dset, '*.mel'), recursive=True))
        self.data = []
        for train_path in dataset:
            y = torch.load(train_path).numpy()
            y[y==-np.inf] = -10
            self.data.append(y)        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        mel = self.data[index]

        # Обрезаем до одинаковой длины
        if mel.shape[1] < hp.seglen:
            pad_len = hp.seglen - mel.shape[1]
            mel = np.pad(mel, ((0,0), (0,pad_len)), mode='wrap')
        elif mel.shape[1] > hp.seglen:
            start_index = random.randint(0,mel.shape[1] - hp.seglen)
            mel = mel[:,start_index:start_index+hp.seglen]
        return mel
        
        
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
        start_index = 0
        if mel.shape[1] < hp.seglen:
            pad_len = hp.seglen - mel.shape[1]
            mel = torch.nn.functional.pad(mel, (0,pad_len))
            wav = torch.nn.functional.pad(wav, (0,pad_len* hp.hop_length))
        elif mel.shape[1] > hp.seglen:
            start_index = random.randint(0,mel.shape[1] - hp.seglen)
            mel = mel[:,start_index:start_index+hp.seglen]        

        start_index *= hp.hop_length
        wav = wav[start_index : start_index + hp.seglen * hp.hop_length]

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
