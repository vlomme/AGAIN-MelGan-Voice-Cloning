import librosa
import numpy as np

class Hparams:
    mel_checkpoint = './logs/mel_ckpt_51k.pt' # Претренированная melgan модель
    wav_checkpoint = "./logs/wav_ckpt_27k.pt" # Претренированная wavgan модель
    again_checkpoint = "./logs/again_40k.pth" # Претренированная again модель ./logs/again_134k.pth
    data_dir = './data' # Путь к звуковым файлам для тренировки  
    train_dir = './train' # Папка где будут храниться mel и звуковые файлы после предобработки
    eval_dir = './eval' # Папка c валидацией где будут храниться mel и звуковые файлы после предобработки
    save_dir = './logs' # Папка куда сохраняются синтезированные файлы и логи во время обучения
    generate_dir = './gen' # Папка с вашими mel или wav файлами для генерации

    batch_size = 64 # Размер batch_size 
    max_epoch = 1000 # Максимальное число эпох Melgan и WavGan
    again_steps = 200000 # Максимальное число шагов Again
    save_interval = 5000 # Интервал сохранения модели
    log_interval = 50 # Интервал вывода ошибки
    lr = 1e-5 # Скорость обучения
    betas1 = 0.5 # Коэффициент для скользящих средних градиента
    betas2 = 0.9 # Коэффициент для квадрата градиента
    lambda_feat = 10 # Коэффициент при вычислении ошибки градиента
    seglen  = 128 #*hop_length Длина сигнала во время обучения
    format = 'wav' # Формат исходных звуковых файлов wav, flac и т.д.  
    sr = 16000 # sampling rate    
    win_length = 1024  # Размер окна
    n_fft = 1024 # Длина кадра
    hop_length = 256 # На сколько выборок сдвигать окно. Менять нельзя, привязана структура
    n_mels = 80 # Сколько мел сверток(фич). Менять нельзя, привязана структура
    trim = 20 # Уровень для удаления шума
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels) # Мел фильтры
    _inv_mel_filtr = np.linalg.pinv(mel_basis) # Обратные мел фильтры

    
    
    
    
    