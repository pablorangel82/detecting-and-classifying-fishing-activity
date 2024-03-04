import utm
from keras.preprocessing.image import ImageDataGenerator
import csv


def load_images(train_dir, val_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_datagen.flow_from_directory(train_dir, class_mode='categorical')

    val_datagen = ImageDataGenerator(rescale=1 / 255)
    val_generator = val_datagen.flow_from_directory(val_dir, class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1 / 255)
    test_generator = test_datagen.flow_from_directory(test_dir, shuffle=False, class_mode='categorical')

    return train_generator, val_generator, test_generator


def get_image_data(file):
    initial_index = file.index('[')
    final_index = file.index(']')
    values = file[initial_index + 1:final_index]
    values = values.split(',')
    return str(values[0]), float(values[1]), float(values[2])


def load_fishing_spots():
    drifting_longlines_spots = []
    fixed_gear = []
    purse_seines_spots = []
    trawlers_spots = []
    with (open('data/fishing_spots.csv', 'r') as csvfile):
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)  # skipping header
        for row in plots:
            classe = row[2]
            if classe == 'unknown':
                continue
            lat_lon = row[4]
            initial_index = lat_lon.index('[')
            final_index = lat_lon.index(']')
            values = lat_lon[initial_index + 2:final_index - 1]
            lat, lon = values.split(',')
            distance = float(row[5])
            vetor = drifting_longlines_spots
            if classe == 'purse_seines':
                vetor = purse_seines_spots
            if classe == 'fixed_gear':
                vetor = fixed_gear
            if classe == 'trawlers':
                vetor = trawlers_spots
            xyCoord = utm.from_latlon(float(lat), float(lon))
            x = xyCoord[0] / 1852
            y = xyCoord[1] / 1852
            area = [x, y, distance]
            vetor.append(area)
        return drifting_longlines_spots, fixed_gear, purse_seines_spots, trawlers_spots
