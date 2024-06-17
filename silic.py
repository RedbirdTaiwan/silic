# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, torch, cv2, os, time, shutil, sys, argparse, mimetypes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from pydub import AudioSegment, effects, scipy_effects
from pydub.utils import mediainfo
from nnAudio import features
import scipy.signal as signal
from yolov5.models.experimental import attempt_load
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from PIL import ImageFont, ImageDraw, Image

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    print(sound_with_altered_frame_rate.frame_rate)
    return sound_with_altered_frame_rate.set_frame_rate(int(sound.frame_rate*speed))

def AudioStandarize(audio_file, sr=32000, device=None, high_pass=0, ultrasonic=False):
  if not device:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  filext = audio_file[-3:].lower()
  if filext == "mp3":
      sound = AudioSegment.from_mp3(audio_file)
  elif filext == "ogg":
      sound = AudioSegment.from_ogg(audio_file)
  elif filext == "wav":
      sound = AudioSegment.from_wav(audio_file)
  else:
      try:
          info = mediainfo(audio_file)
          if info['codec_tag_string'] in ['mp4a']:
              sound = AudioSegment.from_file(audio_file, "m4a")
          else:
              sound = AudioSegment.from_file(audio_file, filext)
      except:
        print('Sorry, this file type is not permitted. The legal extensions are: wav, mp3, wma, m4a, ogg.')
        return None
  original_metadata = {'channel': sound.channels, 'sample_rate':sound.frame_rate, 'sample_size':len(sound.get_array_of_samples()), 'duration':sound.duration_seconds}
  print('Origional audio: channel = %s, sample_rate = %s Hz, sample_size = %s, duration = %s s' %(original_metadata['channel'], original_metadata['sample_rate'], original_metadata['sample_size'], original_metadata['duration']))
  if ultrasonic:
      if sound.frame_rate > 100000: # UltraSonic
          sound = speed_change(sound, 1/12)
      else:
          return False
  if sound.frame_rate > sr:
      sound = scipy_effects.low_pass_filter(sound, sr/2)
  if sound.frame_rate != sr:
      sound = sound.set_frame_rate(sr)
  if sound.channels > 1:
      sound = sound.split_to_mono()[0]
  if not sound.sample_width == 2:
      sound = sound.set_sample_width(2)
  if high_pass:
    sound = sound.high_pass_filter(high_pass)
  sound = effects.normalize(sound) # normalize max-amplitude to 0 dB
  songdata = np.array(sound.get_array_of_samples())
  duration = round(songdata.shape[0]/sound.frame_rate*1000) #ms
  audiodata = torch.tensor(songdata, device=device).float()
  print('Standarized audio: channel = %s, sample_rate = %s Hz, sample_size = %s, duration = %s s' %(sound.channels, sound.frame_rate, songdata.shape[0], sound.duration_seconds))
  return sound.frame_rate, audiodata, duration, sound, original_metadata

def get_media_files(directory):
  media_files = []

  for filename in os.listdir(directory):
    # Get the full path of the file
    filepath = os.path.join(directory, filename)

    # Guess the MIME type of the file
    mime_type, _ = mimetypes.guess_type(filepath)

    if mime_type is not None:
      # If the MIME type is audio or video, add the filename to the list
      if mime_type.startswith('audio') or mime_type.startswith('video'):
        media_files.append(filename)

  return media_files

class Silic:
  """
    Arguments:
        sr (int): Sampling Rate in Hz
        n_fft (int): Window(Frame) Size in samples
        hop_length (str): Frame Step (or Hop Size) in samples
        n_mels (int): The number of Mel filter banks
        fmin (int): The starting frequency for the lowest Mel filter bank in Hz
        fmax (int): The ending frequency for the highest Mel filter bank in Hz
        clip_length (int): The duration of each inference in ms
  """
  def __init__(self, sr=32000, n_fft=1600, hop_length=400, n_mels=240, fmin=100, fmax=15000, device=None, clip_length=3000):
    self.sr = sr
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.n_mels = n_mels
    self.fmin = fmin
    self.fmax = fmax
    self.clip_length = clip_length
    if device:
      self.device = device
    else:
      self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.spec_layer = features.STFT(sr=sr, n_fft=n_fft, hop_length=hop_length).to(self.device)
    self.spec_mel_layer = features.MelSpectrogram(sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, window='hann', center=True, pad_mode='reflect', power=2.0, htk=False, fmin=fmin, fmax=fmax, norm=1, verbose=True).to(self.device)
    self.rainbow_img = torch.tensor([], dtype=torch.float32, device=self.device)
    self.model_path = None
    self.model = None
    self.names = None
    self.soundclasses = None
  
  def audio(self, audio_file, ultrasonic=False):
    self.audiofilename = os.path.basename(audio_file)
    self.audiofilename_without_ext = os.path.splitext(self.audiofilename)[0]
    self.audiopath = os.path.dirname(audio_file)
    self.audiofileext = audio_file.split('.')[-1]
    self.sr, self.audiodata, self.duration, self.sound, self.original_metadata = AudioStandarize(audio_file, self.sr, self.device, high_pass=self.fmin, ultrasonic=ultrasonic)
    self.original_sound = AudioSegment.from_file(audio_file)

  def save_standarized(self, targetmp3path=None):
    if not targetmp3path:
      targetmp3path = os.path.join(self.audiopath, 'mp3', '%s.mp3'%self.audiofilename_without_ext)
      if not os.path.isdir(os.path.dirname(targetmp3path)):
        os.makedirs(os.path.dirname(targetmp3path))
    self.sound.export(targetmp3path, bitrate="128k", format="mp3")
    print('Standarized audio was saved to %s' %targetmp3path)
    return targetmp3path
    
  def spectrogram(self, audiodata, spect_type='linear', rainbow_bands=5):
    if spect_type in ['mel', 'rainbow']:
      spec = self.spec_mel_layer(audiodata)
      w = spec.size()[2]/55
      h = spec.size()[1]/55
      if spect_type == 'mel':
        fig = plt.figure(figsize=(w, h), dpi=100)
        data = torch.sqrt(torch.sqrt(torch.abs(spec[0]) + 1e-6)).cpu().numpy()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.imshow(data, origin='lower', cmap='gray_r', aspect='auto')
      elif rainbow_bands > 1:
        fig, ax = plt.subplots(rainbow_bands, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(w, h))
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        data = torch.log(torch.log(spec[0] + 1e-6))
        for i in range(rainbow_bands):
          subdata = data[i*int(self.n_mels/rainbow_bands):(i+1)*int(self.n_mels/rainbow_bands)].cpu().numpy()
          ax[rainbow_bands-i-1].set_axis_off()
          ax[rainbow_bands-i-1].pcolormesh(subdata, cmap=ListedColormap(cm.rainbow(np.linspace((i+1)/rainbow_bands, (i/rainbow_bands), 32))), rasterized=True)
      else:
        print('Bins of Rainbow should larger than 0.')
        return False
    else:
      spec = self.spec_layer(audiodata)
      data = torch.sqrt(torch.sqrt(torch.abs(spec[0]) + 1e-6)).cpu().numpy()[:,:,0]
      w = data.shape[1]/100*(5/4)*2
      h = data.shape[0]/100*(1/4)*2
      fig = plt.figure(figsize=(w, h), dpi=100)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
      plt.imshow(data, origin='lower', cmap='gray_r', aspect='auto')
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2_img = img #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    
    return cv2_img

  def tfr(self, targetfilepath=None, spect_type='linear', rainbow_bands=5, start=0, stop=None):
    if self.clip_length and ((self.audiodata.size()[0] / self.sr * 1000) < self.clip_length):
        self.audiodata = torch.cat((self.audiodata, torch.zeros(round(self.clip_length*self.sr/1000)-self.audiodata.size()[0], device=self.device)), 0)
    if not stop:
        stop = self.duration
    max_sample_size = 1920000
    if not targetfilepath:
      targetfilepath = os.path.join(self.audiopath, spect_type, '%s.png'%self.audiofilename_without_ext)
      if not os.path.isdir(os.path.dirname(targetfilepath)):
        os.makedirs(os.path.dirname(targetfilepath))
    if not os.path.isdir(os.path.dirname(targetfilepath)):
      print('Error! Cannot find the target folder %s.' %os.path.dirname(targetfilepath))
      exit()
    if (stop - start)/1000*self.sr > (max_sample_size):
        if not os.path.exists('tmp'):
            try:
                os.makedirs('tmp')
            except:
                print('Cannot create tmp folder!')
                exit()
        
        imgs = []
        for ts in range(int(round(start/1000*self.sr)), int(round(stop/1000*self.sr)-self.sr*0.1), max_sample_size):
            if ts+max_sample_size > round(stop/1000*self.sr):
              data = self.audiodata[ts:round(stop/1000*self.sr)+1]
            else:
              data = self.audiodata[ts:ts+max_sample_size]
            try:
              imgs.append(self.spectrogram(data, spect_type, rainbow_bands=rainbow_bands))
            except:
              print('error in converting')
              exit()
        self.cv2_img = cv2.hconcat(imgs)
    else:
        self.cv2_img = self.spectrogram(self.audiodata[int(round(start/1000*self.sr)):int(round(stop/1000*self.sr))], spect_type, rainbow_bands=rainbow_bands)
    
    if spect_type == 'rainbow' and rainbow_bands == 5:
      self.rainbow_img = cv2.cvtColor(self.cv2_img, cv2.COLOR_RGB2BGR)
    
    height, width, colors = self.cv2_img.shape
    #cv2.imwrite(targetfilepath, self.cv2_img)
    PILimage = Image.fromarray(self.cv2_img)
    try:
      PILimage.save(targetfilepath, dpi=(72,72))
    except:
      targetfilepath = '%spng' %targetfilepath[:-3]
      PILimage.save(targetfilepath, dpi=(72,72))
    print('Spectrogram was saved to %s.'%targetfilepath)
    return targetfilepath

  def mel_to_freq(self, mel):
    if mel < 0:
      return self.fmin
    mel = mel*(1127*np.log(1+self.fmax/700)-1127*np.log(1+self.fmin/700)) + 1127*np.log(1+self.fmin/700)
    return round((700*(np.exp(mel/1127)-1)).astype('float32'))

  def xywh2ttff(self, xywh):
    x, y, w, h = list(xywh)
    ts = round((x-w/2)*self.clip_length)
    te = round((x+w/2)*self.clip_length)
    fl = self.mel_to_freq(1-(y+h/2))
    fh = self.mel_to_freq(1-(y-h/2))
    return [ts, te, fl, fh]

  def detect(self, weights, step=1000, conf_thres=0.1, imgsz=640, targetfilepath=None, iou_thres=0.25, targetclasses=None):
    if self.model and self.model_path == weights:
      pass
    else:
      self.model_path = weights
      model = attempt_load(self.model_path, device=self.device)
      self.names = model.module.names if hasattr(model, 'module') else model.names
      model.float()
      self.model = model
      self.soundclasses = pd.read_csv(self.model_path.replace('best.pt', 'soundclass.csv'), encoding='utf8', index_col='sounclass_id').T.to_dict()
    if targetclasses:
      try:
        classes = [self.names.index(name) for name in targetclasses]
      except:
        classes = []
        for i in self.names.keys():
          if self.names[i] in targetclasses:
            classes.append(i)
    else:
      classes = None
    self.tfr(targetfilepath=targetfilepath, spect_type='rainbow')
    
    # prepare input data clips
    dataset = []
    for ts in range(0, self.duration, step):
      clip_start = round(ts/self.duration*self.rainbow_img.shape[1])
      clip_end = clip_start+round(self.clip_length/self.duration*self.rainbow_img.shape[1])
      if clip_end > self.rainbow_img.shape[1]:
        _silence = np.full((self.rainbow_img.shape[0],clip_end-self.rainbow_img.shape[1],3),255).astype('float32')
        _rainbow_img = np.append(self.rainbow_img,_silence,axis=1)
        img0 = _rainbow_img[:,clip_start:clip_end]
        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        dataset.append([os.path.join(self.audiopath, self.audiofilename), img, img0, ts])
        break
      img0 = self.rainbow_img[:,clip_start:clip_end]
      img = letterbox(img0, new_shape=imgsz)[0]
      # Convert
      img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
      img = np.ascontiguousarray(img)
      dataset.append([os.path.join(self.audiopath, self.audiofilename), img, img0, ts])
    
    
    labels = [['file', 'classid', 'species_name', 'sound_class', 'scientific_name', "time_begin", "time_end", "freq_low", "freq_high", "score", "average_power_density", "SNR"]]
    for path, img, im0, time_start in dataset:
      img = torch.from_numpy(img).float().to(self.device)
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
        img = img.unsqueeze(0)
      # Inference
      pred = self.model(img, augment=False)[0]
      pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes)
      for det in pred:    # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]    # normalization gain whwh
        if len(det):
          det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
          for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()    # normalized xywh
            ttff = self.xywh2ttff(xywh)
            ts, te, fl, fh = ttff
            classid = self.names[int(cls)]
            species_name = self.soundclasses[classid]['species_name']
            sound_class = self.soundclasses[classid]['sound_class']
            scientific_name = self.soundclasses[classid]['scientific_name']
            audio = AudioSegment.from_file(os.path.join(self.audiopath, self.audiofilename))
            if audio.channels == 2:
                audio = audio.split_to_mono()[0]  # 轉換為單聲道
            average_power_density_dbfs, snr_db = signal_power(audio, ts/1000, te/1000, fl, fh)
            labels.append([path, classid, species_name, sound_class, scientific_name, round(time_start+ts), round(time_start+te), fl, fh, round(float(conf),3), average_power_density_dbfs, snr_db])
    
    return labels
    
def signal_power(audio, start_time, end_time, low_freq, high_freq):
  sr = audio.frame_rate

  # 取得 bit depth
  bit_depth = audio.sample_width * 8

  # 將音訊轉換為 NumPy 陣列
  samples = np.array(audio.get_array_of_samples())

  # 設定特定時間範圍（以秒為單位）
  start_sample = int(start_time * sr)
  end_sample = int(end_time * sr)
  y_segment = samples[start_sample:end_sample]

  # 設定特定頻率範圍（例如 300-3000 Hz）
  nyquist = 0.5 * sr
  low = low_freq / nyquist
  high = high_freq / nyquist

  # 設計帶通濾波器
  b, a = signal.butter(4, [low, high], btype='band')
  filtered_signal = signal.lfilter(b, a, y_segment)

  # 計算功率密度譜
  f, Pxx = signal.welch(y_segment, sr, nperseg=1024)

  # 選擇特定頻率範圍內的功率密度
  freq_mask = (f >= low_freq) & (f <= high_freq)
  Pxx_in_band = Pxx[freq_mask]

  # 計算特定頻率範圍內的平均功率密度
  average_power_density = np.mean(Pxx_in_band)

  # 根據 bit depth 計算最大可能振幅
  max_possible_amplitude = 2 ** (bit_depth - 1)

  # 將平均功率密度轉換為 dB FS
  average_power_density_dbfs = 10 * np.log10(average_power_density / (max_possible_amplitude ** 2))

  # 計算訊號能量
  signal_power = np.mean(filtered_signal**2)

  # 假設噪聲（這裡用信號減去濾波後的信號作為噪聲估計）
  noise = y_segment - filtered_signal
  noise_power = np.mean(noise**2)

  # 計算 SNR（以分貝為單位）
  snr_db = 10 * np.log10(signal_power / noise_power)
  return round(average_power_density_dbfs, 1), round(snr_db, 1)
  
def get_iou(bb1, bb2):
  """
  © https://github.com/MartinThoma/algorithms/blob/master/CV/IoU/IoU.py
  Calculate the Intersection over Union (IoU) of two bounding boxes.
  Parameters
  ----------
  bb : dict
      Keys: {'x1', 'x2', 'y1', 'y2'}
      The (x1, y1) position is at the top left corner,
      the (x2, y2) position is at the bottom right corner

  Returns
  -------
  float
      in [0, 1]
  """
  assert bb1['x1'] < bb1['x2']
  assert bb1['y1'] < bb1['y2']
  assert bb2['x1'] < bb2['x2']
  assert bb2['y1'] < bb2['y2']

  # determine the coordinates of the intersection rectangle
  x_left = max(bb1['x1'], bb2['x1'])
  y_top = max(bb1['y1'], bb2['y1'])
  x_right = min(bb1['x2'], bb2['x2'])
  y_bottom = min(bb1['y2'], bb2['y2'])

  if x_right < x_left or y_bottom < y_top:
      return 0.0, 0.0, 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
  bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  i_ration_bb1 = intersection_area / bb1_area
  i_ration_bb2 = intersection_area / bb2_area
  assert iou >= 0.0
  assert iou <= 1.0
  return iou, i_ration_bb1, i_ration_bb2

def merge_boxes(bb1, bb2):
  x1 = bb1['x1']
  x2 = bb1['x2']
  y1 = bb1['y1']
  y2 = bb1['y2']
  if bb2['x1'] < bb1['x1']:
    x1 = bb2['x1']
  if bb2['x2'] > bb1['x2']:
    x2 = bb2['x2']
  if bb2['y1'] < bb1['y1']:
    y1 = bb2['y1']
  if bb2['y2'] > bb1['y2']:
    y2 = bb2['y2']
  return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}

def clean_multi_boxes(audiofile, labels, threshold_iou=0.1, threshold_iratio=0.25):
  df = pd.DataFrame(labels[1:],columns=labels[0])
  df = df.sort_values('time_begin')
  df_results = pd.DataFrame()
  soundclasses = df['classid'].unique()
  audio = AudioSegment.from_file(audiofile)
  if audio.channels == 2:
      audio = audio.split_to_mono()[0]  # 轉換為單聲道
  for classid in soundclasses:
    df_class = df[df['classid']==classid].reset_index(drop=True)
    for i in range(0, df_class.shape[0]):
      check = True
      bb1 = {'x1':df_class.loc[i, 'time_begin'], 'x2':df_class.loc[i, 'time_end'], 'y1':df_class.loc[i, 'freq_low'], 'y2':df_class.loc[i, 'freq_high']}
      score1 = df_class.loc[i, 'score']
      j = 0
      for j in range(i+1, df_class.shape[0]):
        bb2 = {'x1':df_class.loc[j, 'time_begin'], 'x2':df_class.loc[j, 'time_end'], 'y1':df_class.loc[j, 'freq_low'], 'y2':df_class.loc[j, 'freq_high']}
        score2 = df_class.loc[j, 'score']
        iou, i_ration_bb1, i_ration_bb2 = get_iou(bb1, bb2)
        i_ration = i_ration_bb1 if i_ration_bb1 > i_ration_bb2 else i_ration_bb2
        if iou >= threshold_iou or i_ration > threshold_iratio:
          score = df_class.loc[i, 'score']
          if df_class.loc[j, 'score'] > score:
            score = df_class.loc[j, 'score']
          merge_box = merge_boxes(bb1, bb2)
          average_power_density, SNR = signal_power(audio, merge_box['x1']/1000, merge_box['x2']/1000, merge_box['y1'], merge_box['y2'])
          try:
            df_class.loc[j, 'time_begin'] = merge_box['x1']
            df_class.loc[j, 'time_end'] = merge_box['x2']
            df_class.loc[j, 'freq_low'] = merge_box['y1']
            df_class.loc[j, 'freq_high'] = merge_box['y2']
            df_class.loc[j, 'score'] = score
            df_class.loc[j, 'average_power_density'] = average_power_density
            df_class.loc[j, 'SNR'] = SNR
          except:
            print(j, df_class.iloc[j])
          check = False
          break
      if check:
        if df_results.shape[0] > 0:
          df_results = pd.concat([df_results, df_class[df_class.index == i]],axis=0, ignore_index=True) 
        else:
          df_results = df_class[df_class.index == i]
  return df_results.sort_values('time_begin').reset_index(drop=True)

def draw_labels(silic, labels, outputpath=None):
  if outputpath and os.path.isdir(outputpath):
    targetpath = os.path.join(outputpath, '%s.png'%silic.audiofilename_without_ext)
  else:
    if not os.path.isdir(os.path.join(silic.audiopath, 'labels')):
      os.makedirs(os.path.join(silic.audiopath, 'labels'))
    targetpath = os.path.join(silic.audiopath, 'labels', '%s.png'%silic.audiofilename_without_ext)
  outputimage = silic.tfr()
  img_pil = Image.open(outputimage)
  width, height = img_pil.size
  fontpath = "model/wt011.ttf"
  font = ImageFont.truetype(fontpath, 9)
  draw = ImageDraw.Draw(img_pil)
  for index, label in labels.iterrows():
    x1 = round(label['time_begin']/silic.duration*width)
    x2 = round(label['time_end']/silic.duration*width)
    y1 = round((1-label['freq_high']/(silic.sr/2))*height)
    y2 = round((1-label['freq_low']/(silic.sr/2))*height)
    tag = '%s%s(%.3f)' %(label['species_name'], label['sound_class'], label['score'])
    draw.text((x1, y1-12),  tag, font = font, fill = 'red')
    draw.rectangle(((x1, y1), (x2, y2)), outline='red')
  try:
    img_pil.save(targetpath)
  except:
    targetpath = '%spng' %targetpath[:-3]
    img_pil.save(targetpath)
  #img_pil.show()
  print(targetpath, 'saved')
  return targetpath


def browser(source, model='', step=1000, targetclasses='', conf_thres=0.1, savepath='result_silic', zip=False):
  if not model:
    for item in os.listdir('model'):
        if os.path.isdir('model/%s'%item):
            model = item
    print(f"Model {model} used.")
  weights=f'model/{model}/best.pt'
  if not targetclasses:
    targetclasses = []
  else:
    targetclasses = [int(item) for item in targetclasses.split(',')]
  t0 = time.time()
  # init
  if savepath and os.path.isdir(savepath):
    result_path = savepath
  else:
    result_path = 'result_silic'
  if os.path.isdir(source) and source == savepath:
    audio_path = None
  else:
    audio_path = os.path.join(result_path, 'audio')
  linear_path = os.path.join(result_path, 'linear')
  rainbow_path = os.path.join(result_path, 'rainbow')
  lable_path = os.path.join(result_path, 'label')
  js_path = os.path.join(result_path, 'js')
  #if os.path.isdir(result_path):
  #  shutil.rmtree(result_path, ignore_errors=True)
  if not os.path.isdir(result_path):
    os.makedirs(result_path)
  if audio_path and not os.path.isdir(audio_path):
    os.makedirs(audio_path)
  if not os.path.isdir(linear_path):
    os.makedirs(linear_path)
  if not os.path.isdir(rainbow_path):
    os.makedirs(rainbow_path)
  if not os.path.isdir(lable_path):
    os.makedirs(lable_path)
  if not os.path.isdir(js_path):
    os.makedirs(js_path)
  shutil.copyfile('browser/index.html', os.path.join(result_path, 'index.html'))
  all_labels = pd.DataFrame()
  model = Silic()
  audiofile = None
  if os.path.isfile(source):
    sourthpath = ''
    audiofiles = [source]
  elif os.path.isdir(source):
    sourthpath = source
    audiofiles = get_media_files(source)
    print(len(audiofiles), 'files found.')
  else:
    print('Files not found')
    exit()
  i = 0
  for audiofile in audiofiles:
    audiofile = os.path.join(sourthpath, audiofile)
    if not audiofile.split('.')[-1].lower() in ['mp3', 'wma', 'm4a', 'ogg', 'wav', 'mp4', 'wma', 'aac']:
      continue
    try:
      model.audio(audiofile)
    except Exception as e:
      print('Error when reading %s'%audiofile)
      print(str(e))
      continue
    i += 1
    if audio_path:
      shutil.copyfile(audiofile, os.path.join(audio_path, model.audiofilename))
    model.tfr(targetfilepath=os.path.join(linear_path, model.audiofilename_without_ext+'.png'))
    labels = model.detect(weights=weights, step=step, targetclasses=targetclasses, conf_thres=conf_thres, targetfilepath=os.path.join(rainbow_path, model.audiofilename_without_ext+'.png'))
    if len(labels) == 1:
      print("No sound found in %s." %audiofile)
    else:
      newlabels = clean_multi_boxes(audiofile, labels)
      newlabels['file'] = model.audiofilename
      newlabels.to_csv(os.path.join(lable_path, model.audiofilename_without_ext+'.csv'), index=False, encoding='utf-8-sig')
      if all_labels.shape[0] > 0:
        all_labels = pd.concat([all_labels, newlabels],axis=0, ignore_index=True) 
      else:
        all_labels = newlabels
      print("%s sounds of %s species is/are found in %s" %(newlabels.shape[0], len(newlabels['classid'].unique()), audiofile))

  if all_labels.shape[0] == 0:
    print('No sounds found!')
  else:
    all_labels.to_csv(os.path.join(lable_path, 'labels.csv'), index=False, encoding='utf-8-sig')
    print('%s sounds of %s species is/are found in %s recording(s). Preparing the browser package ...' %(all_labels.shape[0], len(all_labels['classid'].unique()), i))
    df_classes = pd.read_csv(weights.replace('best.pt', 'soundclass.csv'))
    if targetclasses:
      df_classes = df_classes[df_classes['sounclass_id'].isin(targetclasses)]
    else:
      names = all_labels['classid'].unique()
      df_classes = df_classes[df_classes['sounclass_id'].isin(names)]
    with open(os.path.join(js_path, 'soundclass.js'), 'w', newline='', encoding='utf-8') as csv_file:
      csv_file.write('var sounds = { \n')
      for index, row in df_classes.iterrows():
        csv_file.write('"%s": ["%s", "%s", "%s"], \n' %(row['sounclass_id'], row['species_name'], row['sound_class'], row['scientific_name']))
      csv_file.write('};')

    with open(os.path.join(js_path, 'labels.js'), 'w', newline='', encoding='utf-8') as f:
      f.write('var  labels  =  [' + '\n')
      for index, label in all_labels.iterrows():
        f.write("['{}', {}, {}, {}, {}, {}, {}],\n".format(label['file'].replace("'", "\\'"), label['time_begin'], label['time_end'], label['freq_low'], label['freq_high'], label['classid'], label['score']))
      f.write('];' + '\n')
    
    if zip:
        shutil.make_archive('result_silic', 'zip', result_path)
        print('Finished. The browser package is compressed and named result_silic.zip')
    else:
        print('Finished. All results were saved in the folder %s' %result_path)
    print(time.time()-t0, 'used.')

def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, help='Source of a single file or 1-level folder')
  parser.add_argument('--model', type=str, default="", help='Version of model wights')
  parser.add_argument('--step', type=int, default=1000, help='Length of sliding window in ms.')
  parser.add_argument('--targetclasses', type=str, default='', help='filter by class, comma-separated')
  parser.add_argument('--conf_thres', type=float, default=0.1, help='Threshold of confidence score from 0.0 to 1.0')
  parser.add_argument('--savepath', type=str, default='result_silic', help='Target folder of inference results archived')
  parser.add_argument('--zip', action='store_true', help='ZIP')
  opt = parser.parse_args()
  return opt

if __name__ == '__main__':
  opt = parse_opt()
  browser(**vars(opt))
