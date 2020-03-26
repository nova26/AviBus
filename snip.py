import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sqlite3

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import math
from datetime import datetime

from sklearn.utils import shuffle

import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers, utils, models
from sklearn.metrics import classification_report, confusion_matrix

HOME_BUS_LAT_LONG = (math.radians(32.176398), math.radians(34.903889))
BUS_HOME_LAT_LONG = (math.radians(32.1783219), math.radians(34.9268141))

dayToNum = {"Sunday": 1,
            "Monday": 2,
            "Tuesday": 3,
            "Wednesday": 4,
            "Thursday": 5,
            "Friday": 6,
            "Saturday": 7}


def getMaxDeltaTime(fileName):
    with open(fileName, 'r') as in_file:
        max1 = 0
        time = ''
        in_file.readline()
        in_file.readline()
        prevLine = list()
        for line in in_file:
            line = line.replace("\n", "")
            line = line.split(",")

            if line[-1] == '-1.0' and prevLine[-1] != '-1.0':
                expectedMin = int((prevLine[-3].split(":")[1]))
                nowMin = int((prevLine[0].split(":")[1]))
                expectedHour = int((prevLine[-3].split(":")[0]))
                nowHour = int((prevLine[0].split(":")[0]))
                if expectedHour != nowHour:
                    if expectedHour > nowHour:
                        delta = (60 - nowMin) + expectedMin
                        print(prevLine[0] + " Delta " + str(delta))
                        if delta > max1:
                            max1 = delta
                            time = prevLine[0]
                    else:
                        print("ERROR " + str(line))
                else:
                    print(prevLine[0] + " Delta " + str(abs(expectedMin - nowMin)))
                    if abs(expectedMin - nowMin) > max1:
                        max1 = abs(expectedMin - nowMin)
                        time = prevLine[0]
            prevLine = line

        print("Time " + time)
        print("Max " + str(max1))


def getXYFromCsv(fileName):
    with open(fileName, 'r') as the_file:
        xList = list()
        yList = list()
        isFirst = True
        for line in the_file:
            line = line.split(",")
            line[1] = line[1].replace("\n", "")
            if isFirst:
                isFirst = False
                continue
            xList.append(float(line[0]))
            yList.append(float(line[1]))
    return xList, yList


def plotCsv(fileName):
    latList, lonList = getXYFromCsv(fileName)
    plt.scatter(latList, lonList)
    plt.title(fileName)
    plt.xlabel("Lat")
    plt.ylabel("Lon")
    plt.show()


def getBusLineData(busline):
    DB_ROOT = "C:\Databases\\bus.db"
    conn = sqlite3.connect(DB_ROOT)
    c = conn.cursor()
    lineNumber = (busline,)
    c.execute('SELECT * FROM tbl_kfar_saba WHERE line=?', lineNumber)
    results = c.fetchall()
    return results


def getBusLineLatLong(busNumber):
    DB_ROOT = "C:\Databases\\bus.db"
    conn = sqlite3.connect(DB_ROOT)
    c = conn.cursor()
    lineNumber = (busNumber,)
    c.execute('SELECT * FROM tbl_kfar_saba_data WHERE  DepartureMonth=10 AND line=?', lineNumber)
    results = c.fetchall()
    latList = list()
    lonList = list()

    for ent in results:
        lat = ent[-2]
        lon = ent[-1]
        if lat == -1:
            continue
        latList.append(lat)
        lonList.append(lon)
    return latList, lonList


def plotBusLineLocations(busline):
    latList, lonList = getBusLineLatLong(busline)
    plt.scatter(latList, lonList)
    plt.xlabel("Lat")
    plt.ylabel("Lon")
    plt.show()


def writeCSVBusLineRawData(busline, fileName):
    with open(fileName, mode='w', newline='') as the_file:
        writer = csv.writer(the_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['timeStamp', 'line', 'departureYear', 'departureMonth', 'departureDay', 'expectedYear', 'expectedMonth',
             'expectedDay', 'latitude', 'longitude'])
        busData = getBusLineData(busline)
        for ent in busData:
            writer.writerow(
                [ent[0], ent[1], ent[2], ent[3], ent[4], ent[5], ent[6], ent[7], ent[8], ent[9], ent[10], ent[11]])


def drop_numerical_outliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='reduce') \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)


def latLongDistanse(pointA, pointB):
    deltaPhi = abs(pointA[0] - pointB[0])
    deltaLambda = abs(pointA[1] - pointB[1])
    a = pow(math.sin(deltaPhi / 2), 2) + math.cos(pointA[0]) * math.cos(pointB[0]) * pow(math.sin(deltaLambda / 2), 2)
    c = math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = 6371 * c * 1000
    return d


def writeConvertLatLongToDistance(inputfile):
    with open(inputfile, 'r') as in_file:
        outf = inputfile.replace(".csv", "_distance.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['timeStamp', 'departureDay', 'distance'])
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                latitude = math.radians(float(line[-2]))
                longitude = math.radians(float(line[-1]))
                t = (latitude, longitude)
                distanse = int(latLongDistanse(t, HOME_BUS_LAT_LONG))
                lineDate = line[2] + "-" + line[3] + "-" + line[4]
                df = None
                try:
                    df = pd.Timestamp(lineDate)
                except:
                    lineDate = line[2] + "-" + line[4] + "-" + line[3]
                    df = pd.Timestamp(lineDate)

                writer.writerow([line[0], df.day_name(), distanse])


def wrtieFileLatLong(fileName):
    freshFile = fileName.replace(".csv", "_ll.csv")

    with open(freshFile, mode='w', newline='') as out_file:
        with open(fileName, mode='r') as in_file:
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                out_file.write(line[-2] + "," + line[-1] + "\n")


def plotLatLong(fileName):
    boston_df = pd.read_csv(fileName)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(boston_df['timeStamp'], boston_df['distance'])
    ax.set_xlabel('Proportion of non-retail business acres per town')
    ax.set_ylabel('Full-value property-tax rate per $10,000')
    plt.show()


def writeBusTime(fileName):
    with open(fileName.replace(".csv", "_arrivalTime.csv"), mode='w', newline='') as out_file:
        out_file.write("timeStamp,departureDay,distance,arrivalTime\n")
        with open(fileName, mode='r', newline=None) as in_file:
            in_file.readline()
            in_file.readline()
            batch = list()
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                line[-1] = int(line[-1])
                if line[-1] == 2638254:
                    minElement = -1
                    if len(batch) != 0:
                        minElement = min(batch, key=lambda t: t[2])
                    for l in batch:
                        l.append(minElement[0])
                        out_file.write(l[0] + "," + l[1] + "," + str(l[2]) + "," + l[3] + "\n")

                    batch = list()
                else:
                    batch.append(line)


def wrtieBusDeltaTime(fileName):
    with open(fileName.replace(".csv", "_deltaTime.csv"), mode='w', newline='') as out_file:
        out_file.write("timeStamp,departureDay,distance,minutesLeftToArrive\n")
        with open(fileName, mode='r', newline=None) as in_file:
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                timeStame = datetime.strptime(line[0], '%H:%M:%S')
                arrivalTime = datetime.strptime(line[-1], '%H:%M:%S')
                delta = arrivalTime - timeStame
                deltaSec = int(str(delta).split(":")[2])
                deltaMin = int(str(delta).split(":")[1])
                if deltaSec > 30:
                    deltaMin = deltaMin + 1

                out_file.write(line[0] + "," + line[1] + "," + line[2] + "," + str(deltaMin) + "\n")
                pass


def splitTime(fileName):
    with open(fileName.replace(".csv", "_splitTime.csv"), mode='w', newline='') as out_file:
        out_file.write("hour,minute,second,,departureDay,distance,minutesLeftToArrive\n")
        with open(fileName, mode='r', newline=None) as in_file:
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                timeStsmp = line[0].split(":")
                out_file.write(
                    str(int(timeStsmp[0])) + "," + str(int(timeStsmp[1])) + "," + str(int(timeStsmp[2])) + "," + line[
                        1] + "," + line[2] + "," + line[3] + "\n")


def writeConvertDayNameToDayNum(fileName):
    global dayToNum
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_dayNum.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['timeStamp', 'departureDay', 'distance'])
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                lineDate = dayToNum[line[1]]
                writer.writerow([line[0], lineDate, line[2]])


def aggregateSecond(fileName):
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_MinAggre.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['hour', 'minute', 'departureDay', 'distance', 'minutesLeftToArrive'])
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "")
                line = line.split(",")
                line = [int(line[x]) for x in range(len(line))]
                # if line[2]>=30:
                #     line[1]=line[1]+1
                #     if line[1] ==60:
                #         line[1]=0
                #         line[0] = line[0]+1
                writer.writerow([line[0], line[1], line[3], line[4], line[5]])


def writeArrivalTime(fileName):
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_ArrivalTime.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['hour', 'minute', 'departureDay', 'distance', 'arrivaltime'])
            in_file.readline()
            batchLines = []
            prev_time = None
            for line in in_file:
                line = line.replace("\n", "").strip()
                line = line.split(",")
                current_time = datetime.strptime(line[0], '%H:%M:%S')

                batchLines.append(line)

                if prev_time is None:
                    prev_time = current_time
                else:
                    delta = current_time - prev_time
                    delta = int(str(delta).split(":")[1])

                    if delta > 19:
                        distance = 100000
                        time = None
                        for batchLine in batchLines:
                            if int(batchLine[-1]) <= distance:
                                distance = int(batchLine[-1])
                                time = batchLine[0]

                        for l in batchLines:
                            writer.writerow([l[0], l[1], l[2], time])
                        batchLines = []
                        # batchLines.append(line)
                        prev_time = None
                if prev_time is None:
                    continue
                prev_time = current_time


def writeArrivalTimeInMin(fileName):
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_ArrivalTimeMin.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['hour', 'minute', 'departureDay', 'distance', 'arrivaltime'])
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "").strip()
                line = line.split(",")
                dipTime = datetime.strptime(line[0], '%H:%M:%S')
                arriveTime = datetime.strptime(line[-1], '%H:%M:%S')
                delta = arriveTime - dipTime
                if 'day' in str(delta):
                    continue
                delta = str(delta)
                delta = delta.split(":")[1]
                delta = int(delta)
                writer.writerow([line[0], line[1], line[2], delta])


def writeSplitTime(fileName):
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_splitTime.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['hour', 'minute', 'departureDay', 'distance', 'arrivaltime'])
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "").strip()
                line = line.split(",")
                time = line[0].split(':')
                time = [int(t) for t in time]
                writer.writerow([time[0], time[1], line[1], line[2], line[3]])


def padArray(array, size):
    while len(array) != size:
        array.insert(0, 0)
    return array


def OneHotVed(fileName):
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_onehot.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['hour', 'minute', 'departureDay', 'distance', 'arrivaltime'])
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "").strip()
                line = line.split(",")

                hour = int(line[0])
                hour = [int(i) for i in list('{0:0b}'.format(hour))]
                hour = padArray(hour, 5)

                minute = int(line[1])
                minute = [int(i) for i in list('{0:0b}'.format(minute))]
                minute = padArray(minute, 6)

                day = int(line[2])
                day = [int(i) for i in list('{0:0b}'.format(day))]
                day = padArray(day, 6)

                distance = int(line[3])
                if distance > 2000:
                    continue
                distance = [int(i) for i in list('{0:0b}'.format(distance))]
                distance = padArray(distance, 11)

                arrivalTime = int(line[4])

                arrivalTime = [int(i) for i in list('{0:0b}'.format(arrivalTime))]
                arrivalTime = padArray(arrivalTime, 6)

                res = hour + minute + day + distance + arrivalTime
                res = str(res)
                res = res.replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
                writer.writerow(res)


def createDataset(fileName):
    with open(fileName, 'r') as in_file:
        outf = fileName.replace(".csv", "_noextreme.csv")
        with open(outf, mode='w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['hour', 'minute', 'departureDay', 'distance', 'arrivaltime'])
            in_file.readline()

            in_file.readline()
            X_train = np.full((1, 4), 0)
            y_train = np.full((1, 1), 0)
            y_train = []
            in_file.readline()
            for line in in_file:
                line = line.replace("\n", "").strip()
                line = line.split(",")
                data = line[0:-1]

                data = [float(x) for x in data]
                label = int(line[-1])

                if label > 23 or data[-1] > 3000 or data[1] > 24 or data[2] > 5:
                    continue

                data = np.reshape(data, (1, 4))
                X_train = np.concatenate((X_train, data), axis=0)
                y_train.append(label)
                print(X_train.shape[0])

                writer.writerow([data[0][0], data[0][1], data[0][2], data[0][3], label])

            X_train = np.delete(X_train, 0, 0)

            pickle.dump(X_train, open("./data/X_train", "wb"))
            pickle.dump(y_train, open("./data/y_train", "wb"))


def trainModel():
    X_train = pickle.load(open("./data/X_train", "rb"))
    y_train = pickle.load(open("./data/y_train", "rb"))

    for x in range(4):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)

    X_train = np.array(X_train, dtype=np.float64)

    xMean = np.mean(X_train, axis=0)
    xStd = np.std(X_train, axis=0)

    X_train -= xMean
    X_train /= (xStd + 1e-8)

    y_train = utils.to_categorical(y_train, 24)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, test_size=0.33, random_state=42)

    if os.path.isfile('./models/model.h5'):
        print("Building imageSetimagesPath ...")
        model = models.load_model('./models/model.h5')
        y_pred_class = model.predict(X_val)

        y_pred = []
        y_ground = []

        for pred in y_pred_class:
            maxVal = pred.max()
            predVal = np.where(pred == maxVal)[0][0]
            y_pred.append(predVal)

        for pred in y_val:
            maxVal = pred.max()
            predVal = np.where(pred == maxVal)[0][0]
            y_ground.append(predVal)


        cm = confusion_matrix(y_ground, y_pred)
        print(cm)

    model = Sequential([
        Dense(2048, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(units=2048, activation='relu'),
        Dense(units=1024, activation='relu'),
        Dropout(0.5),
        Dense(units=512, activation='relu'),
        Dense(units=512, activation='relu'),
        Dropout(0.5),
        Dense(units=64, activation='relu'),
        Dense(units=64, activation='relu'),
        Dropout(0.5),
        Dense(units=32, activation='relu'),
        Dense(units=32, activation='relu'),
        Dropout(0.5),
        Dense(units=16, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=8, activation='relu'),
        Dense(24, activation='softmax')
    ])

    fileName = 'Model_epoch{epoch:02d}_loss{val_loss:.4f}_acc{val_accuracy:.2f}.h5'

    modelCheckPoint = ModelCheckpoint(
        './models/' + fileName,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1
    )

    earlyStopping = EarlyStopping(
        monitor='val_loss',
        patience=40
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
                                  verbose=1,
                                  mode='auto',
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=0.001)

    callbacks_list = [modelCheckPoint, earlyStopping, reduce_lr]

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #  opt = optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    # opt = optimizers.SGD(lr=0.01, momentum=0.8, nesterov=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, callbacks=callbacks_list,
                        validation_data=(X_val, y_val),
                        epochs=1000,
                        shuffle=True,
                        batch_size=225,
                        verbose=1
                        )

    pd.DataFrame(history.history).plot(figsize=(16, 16))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


# writeCSVBusLineRawData(610, "./data/610_Raw_Data.csv")
# writeConvertLatLongToDistance("./data/610_Raw_Data.csv")
# writeConvertDayNameToDayNum("./data/610_Raw_Data_distance.csv")
# writeArrivalTime("./data/610_Raw_Data_distance_dayNum.csv")
# writeArrivalTimeInMin("./data/610_Raw_Data_distance_dayNum_ArrivalTime.csv")
# writeSplitTime("./data/610_Raw_Data_distance_dayNum_ArrivalTime_ArrivalTimeMin.csv")
# createDataset("./data/610_Data_noextreme_wekaClean.csv")
trainModel()
