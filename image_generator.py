import matplotlib.pyplot as plt
from PIL import Image
import csv
import utm
import os, shutil


def create(classe, flag):
    x = []
    y = []
    v = []

    if flag == 0:
        if not os.path.exists(os.path.dirname('images/not_fishing/')):
            os.makedirs(os.path.dirname('images/not_fishing/'))
    else:
        if os.path.exists(os.path.dirname('images/' + classe+'/')):
            shutil.rmtree('images/' + classe+'/')
        os.makedirs(os.path.dirname('images/' + classe+'/'))
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    with (open('data/' + classe + '.csv', 'r') as csvfile):
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)  # skipping header
        mmsiAnterior = 'none'
        mmsiAtual = 'none'
        cont_train = 0
        if flag == 0:
            classe = 'not_fishing'
        for row in plots:
            mmsiAnterior = mmsiAtual
            mmsiAtual = row[0]
            source = row[9]
            if float(row[8]) != flag or source == 'false_positives' or mmsiAtual != mmsiAnterior:
                if len(x) >= 3:
                    plt.axis('off')
                    for i in range((len(v))):
                        color = 'black'
                        if (v[i] > 8):
                            color = 'blue'
                        else:
                            if (v[i] > 4):
                                color = 'red'
                            else:
                                if (v[i] > 2):
                                    color = 'green'
                                else:
                                    color = 'black'
                        if i < (len(x) - 1):
                            a = []
                            b = []
                            a.append(x[i])
                            a.append(x[i + 1])
                            b.append(y[i])
                            b.append(y[i + 1])
                            plt.plot(a, b, marker='o', color=color)
                    cont_train = cont_train + 1
                    filename = 'images/' + classe + '/' + str(mmsiAtual) + '-' + str(cont_train) + 'A.png'
                    plt.savefig(filename)
                    plt.clf()

                    if classe == 'purse_seines':
                        im = Image.open(filename).convert('RGBA')
                        rotated_img = im.rotate(180)
                        plt.imshow(rotated_img)
                        plt.axis('off')
                        plt.savefig(filename.replace('A', 'B'))
                        plt.clf()

                x.clear()
                y.clear()
                if float(row[8]) != flag or str(row[9]) == 'false_positives':
                    mmsiAnterior = 'none'
                    continue
            lat = float(row[6])
            lon = float(row[7])
            xyCoord = utm.from_latlon(lat, lon)
            currentX = xyCoord[0] / 1852
            currentY = xyCoord[1] / 1852
            x.append(currentX)
            y.append(currentY)
            veloc = float(row[4])
            v.append(veloc)

    path = 'images/' + classe + '/'
    moveto = 'images/'
    files = os.listdir(path)
    cont_train = 0
    cont_test = 0

    if flag == 0:
        if not os.path.exists(os.path.dirname('images/train/not_fishing')):
            os.makedirs(os.path.dirname('images/train/not_fishing'))
        if not os.path.exists(os.path.dirname('images/test/not_fishing')):
            os.makedirs(os.path.dirname('images/test/not_fishing'))
        if not os.path.exists(os.path.dirname('images/val/not_fishing')):
            os.makedirs(os.path.dirname('images/val/not_fishing'))
    else:
        os.makedirs(os.path.dirname('images/train/' + classe+'/'))
        os.makedirs(os.path.dirname('images/test/' + classe + '/'))
        os.makedirs(os.path.dirname('images/val/' + classe + '/'))
    for f in files:
        src = path + f
        type = 'train/'
        if cont_train < len(files) * 0.70:
            type = 'train/'
            cont_train = cont_train + 1
        else:
            if cont_test > ((len(files) * (1 - 0.70)) * 0.5):
                type = 'test/'
            else:
                type = 'val/'
            cont_test = cont_test + 1
        dst = moveto + type + classe + '/' + f
        shutil.move(src, dst)

try:
    shutil.rmtree('images/train')
except:
    print('Error: train folder does not exists')
try:
    shutil.rmtree('images/test')
except:
    print('Error: test folder does not exists')
try:
    shutil.rmtree('images/val')
except:
    print('Error: validation folder does not exists')

try:
    shutil.rmtree('images/not_fishing')
except:
    print('Nothing to delete: \'not fishing\' files was not created before')

create('drifting_longlines', 1)
create('drifting_longlines', 0)
create('purse_seines', 1)
create('purse_seines', 0)
create('trawlers', 1)
create('trawlers', 0)
shutil.rmtree('images/drifting_longlines')
shutil.rmtree('images/purse_seines')
shutil.rmtree('images/trawlers')
shutil.rmtree('images/not_fishing')
print('Terminou')
