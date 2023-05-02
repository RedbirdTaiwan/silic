import tkinter as tk, pandas as pd, time, os
from tkinter import messagebox, HORIZONTAL, filedialog
from silic import *

root = tk.Tk()
root.title('SILIC - Sound Identification and Labeling Intelligence for Creatures')
#root.configure(bg="#847A5C")    #可以直接打顏色名稱或是找色碼表的代號
root.iconbitmap('model/LOGO_circle.ico')

inputfolder = tk.StringVar(root)
outputfolder = tk.StringVar(root)
threshold = tk.DoubleVar(root)
model_weight = tk.StringVar(root)
target_classes = tk.StringVar(root)
w = 800
h = 600
x = round((root.winfo_screenwidth()-w)/2)
y = round((root.winfo_screenheight()-h)/2)
root.geometry(f"{w}x{h}+{x}+{y}")
"""
def show_messagebox():
    # messagebox.showinfo('My messagebox','Hola')
    # messagebox.showwarning('My messagebox','Oops!')
    # messagebox.showerror('My messagebox','Error!!!')
    # messagebox.askokcancel('My messagebox','Cancel or not ?')
    # messagebox.askquestion('My messagebox','Are you sure you want to leave ?')
    # messagebox.askretrycancel('My messagebox','重試或取消?')
    messagebox.askyesnocancel('My messagebox','是或否或取消?')

noLabel = tk.Button(root, text="RUN",bg='#BFBFBF',fg="#000000", font=("Arial", 16, "bold"), relief="raised", command=show_messagebox) #建立raised標籤
noLabel.pack(anchor='se',side='right',padx=10,pady=10)

#image = Image.open("LOGO.png")
#logo = ImageTk.PhotoImage(image)
#label = tk.Label(root, image=logo)
#label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
"""
#def browser(source, model='exp24', step=1000, targetclasses='', conf_thres=0.1, savepath='result_silic', zip=False):
def select_input_folder():
    path = filedialog.askdirectory()
    inputfolder.set(path)
    folder_input_path_label.config(text=path)

def select_output_folder():
    path = filedialog.askdirectory()
    outputfolder.set(path)
    folder_output_path_label.config(text=path)

def filter_options(event):
    # Get current text in entry
    text = filter.get()

    # Clear the listbox
    listbox2.delete(0, tk.END)

    # Refill listbox with filtered options
    for option in options:
        if option.find(text) >= 0:
            listbox2.insert(tk.END, option)

def shift_selection(event):
    # Get selected item
    selected = event.widget.curselection()

    # Get the listbox that fired the event
    source = event.widget

    # Determine the destination listbox
    destination = listbox2 if source == listbox1 else listbox1

    # Move selected item
    for i in selected:
        item = source.get(i)
        destination.insert(tk.END, item)
        source.delete(i)

def setThreshold(source):
    threshold.set(thresholdSlider.get())

def setModel(source):
    classes = readclassfile()
    listbox1.delete(0, tk.END)
    listbox2.delete(0, tk.END)
    options = classes.keys()
    messagebox.showinfo('Model Seletion', 'Model %s was selected including %s sound classes.' %(model_weight.get(), len(options)))
    for item in options:
        listbox2.insert(tk.END, item)

def readclassfile():
    classes = {}
    df = pd.read_csv('model/%s/soundclass.csv'%model_weight.get(), lineterminator='\n', encoding="utf-8")
    df = df.sort_values(by=['species_name', 'sound_class'])
    for index, row in df.iterrows():
        classes["%s: %s(%s) %s"%(row['sounclass_id'], row['species_name'], row['scientific_name'], row['sound_class'])] = {'sounclass_id': row['sounclass_id'], 'species_name': row['species_name'], 'sound_class': row['sound_class'], 'scientific_name':row['scientific_name']}
    return classes

def run():
    if not inputfolder.get():
        messagebox.showwarning('Warning','No input folder found.')
        return False
    if not outputfolder.get():
        messagebox.showwarning('Warning','No output folder found.')
        return False
    targetclasses = []
    if not listbox1.get(0, tk.END):
        #messagebox.showinfo('My messagebox', "All classes are selected.")
        targetclasses = ''
    else:
        for item in listbox1.get(0, tk.END):
            targetclasses.append(str(classes[item]['sounclass_id']))
        #messagebox.showinfo('My messagebox', "%s classes are selected." %len(targetclasses))
        
    media_files = get_media_files(inputfolder.get())
    text.delete("1.0", tk.END)
    text.insert(tk.END, "SILIC Detector: %s files are found, detecting..... Please wait a moment.\n" %len(media_files))
    text.see(tk.END)
    root.update()
    model = model_weight.get()
    source = inputfolder.get()
    savepath = outputfolder.get()
    conf_thres = threshold.get()
    step = 1000
    weights=f'model/{model}/best.pt'
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
    text.insert(tk.END, "Loading SILIC ...\n")
    text.see(tk.END)
    root.update()
    model = Silic()
    text.insert(tk.END, "SILIC loaded. Detecting ...\n")
    text.see(tk.END)
    root.update()
    sourthpath = source
    i = 0
    for audiofile in media_files:
        audiofile = os.path.join(sourthpath, audiofile)
        if not audiofile.split('.')[-1].lower() in ['mp3', 'wma', 'm4a', 'ogg', 'wav', 'mp4', 'wma', 'aac']:
            continue
        model.audio(audiofile)
        i += 1
        if audio_path:
            shutil.copyfile(audiofile, os.path.join(audio_path, model.audiofilename))
        model.tfr(targetfilepath=os.path.join(linear_path, model.audiofilename_without_ext+'.png'))
        labels = model.detect(weights=weights, step=step, targetclasses=targetclasses, conf_thres=conf_thres, targetfilepath=os.path.join(rainbow_path, model.audiofilename_without_ext+'.png'))
        if len(labels) == 1:
            text.insert(tk.END, "No sound found in %s.\n" %audiofile)
            text.see(tk.END)
            root.update()
        else:
            newlabels = clean_multi_boxes(labels)
            newlabels['file'] = model.audiofilename
            newlabels.to_csv(os.path.join(lable_path, model.audiofilename_without_ext+'.csv'), index=False)
        if all_labels.shape[0] > 0:
            all_labels = all_labels = pd.concat([all_labels, newlabels],axis=0, ignore_index=True) 
        else:
            all_labels = newlabels
        text.insert(tk.END, "%s sounds of %s species is/are found in %s\n" %(newlabels.shape[0], len(newlabels['classid'].unique()), audiofile))
        text.see(tk.END)
        root.update()

    if all_labels.shape[0] == 0:
        text.insert(tk.END, 'No sounds found!\n')
        text.see(tk.END)
        root.update()
    else:
        all_labels.to_csv(os.path.join(lable_path, 'labels.csv'), index=False)
        text.insert(tk.END, '%s sounds of %s species is/are found in %s recording(s). Preparing the browser package ...\n' %(all_labels.shape[0], len(all_labels['classid'].unique()), i))
        text.see(tk.END)
        root.update()
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
        
        text.insert(tk.END, 'Finished. All results were saved in the folder %s\n' %result_path)
        text.insert(tk.END, '%s used.' %str(time.time()-t0))
        text.see(tk.END)
        root.update()
        os.startfile(result_path)


root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=2, minsize=240)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=2, minsize=240)

logo = tk.PhotoImage(file="model/silic_logo_full.png")
logo_label = tk.Label(root, image=logo)
logo_label.grid(row=0,column=0,rowspan=2,sticky=tk.N+tk.S+tk.W+tk.E)

select_input_folder_button = tk.Button(root, text="Select Input Folder", command=select_input_folder)
select_input_folder_button.grid(row=0,column=1,sticky=tk.N+tk.S+tk.W+tk.E)

folder_input_path_label = tk.Label(root, text="", anchor='w')
folder_input_path_label.grid(row=0,column=2,columnspan=2, sticky=tk.E+tk.W)

select_output_folder_button = tk.Button(root, text="Select Output Folder", command=select_output_folder)
select_output_folder_button.grid(row=1,column=1,sticky=tk.E+tk.W)

folder_output_path_label = tk.Label(root, text="", anchor='w')
folder_output_path_label.grid(row=1,column=2,columnspan=2,sticky=tk.E+tk.W)

thresholdSlider_label = tk.Label(root, text="Confidence Threshold", anchor='w')
thresholdSlider_label.grid(row=2,column=0,sticky=tk.E+tk.W)

thresholdSlider = tk.Scale (root, from_= 0.0 , to = 1.0 , resolution=0.01, length=200, orient=HORIZONTAL, command=setThreshold)
thresholdSlider.grid(row=2,column=1,sticky=tk.E+tk.W)
thresholdSlider.set(0.1)

model_label = tk.Label(root, text="Model version", anchor='w')
model_label.grid(row=2,column=2,sticky=tk.E+tk.W)

sets = []
latestmodel = ''
for item in os.listdir('model'):
    if os.path.isdir('model/%s'%item):
        sets.append(item)
        latestmodel = item
model_weight.set(latestmodel)
opm=tk.OptionMenu(root, model_weight, *sets, command=setModel)
opm.grid(row=2,column=3,sticky=tk.E+tk.W)
classes = readclassfile()

target = tk.Label(root, text="Target Classes (Left empty when detect all classes)")
target.grid(row=3,column=0,columnspan=2,sticky=tk.E+tk.W)

filter_label = tk.Label(root, text="Class filter")
filter_label.grid(row=3,column=2,sticky=tk.E+tk.W)

filter = tk.Entry(root)
filter.bind('<KeyRelease>', filter_options)
filter.grid(row=3,column=3)

listbox1 = tk.Listbox(root, height=20)
listbox1.bind('<Double-Button-1>', shift_selection)
listbox1.grid(row=4,column=0,columnspan=2,sticky=tk.E+tk.W)

listbox2 = tk.Listbox(root, height=20)
listbox2.bind('<Double-Button-1>', shift_selection)
listbox2.grid(row=4,column=2,columnspan=2,sticky=tk.E+tk.W)

# Populate the right listbox with some data
options = classes.keys()
for item in options:
    listbox2.insert(tk.END, item)

run_button = tk.Button(root, text="RUN",bg='#847A5C',fg="#000000", relief="raised", command=run)
run_button.grid(row=5,column=0,columnspan=5,sticky=tk.E+tk.W)

text = tk.Text(root, height=10)
text.grid(row=6,column=0,columnspan=5,sticky=tk.E+tk.W)

root.mainloop()
