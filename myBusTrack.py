# CTRL+ALT+I
import requests
import datetime
import xml.etree.ElementTree as ET
import time
import sqlite3

from playsound import playsound
import threading
threadCount = 0

print("Start...")

def playSound(index):
    global threadCount

    dateNowObj = str(datetime.datetime.now())
    dateNowObj = int(dateNowObj.split()[1].split(":")[0])

    if dateNowObj>9 or dateNowObj<7:
        return

    if threadCount == 0:
        threadCount = threadCount + 1
        playsound('C:\\Users\\Avi\\Desktop\\PyProj\\AviBus\\sound\\sound.mp3')
        time.sleep(30)
        threadCount = 0



DB_ROOT = "C:\Databases\\bus.db"
WEB_SERVICE_URL = 'http://siri.motrealtime.co.il:8081/Siri/SiriServices'
TABLE_NAME = "tbl_kfar_saba"
KENYON_HARIM = '37030'
ZEHEV = '32837'
BUS_STATION = ZEHEV


def formatDateTime(dateTimeString):
    originAimedDepartureTime = dateTimeString.split()
    originAimedDepartureDate = originAimedDepartureTime[0]
    originAimedDepartureTime = originAimedDepartureTime[1]
    originAimedDepartureDate = originAimedDepartureDate.split("-")
    departureYear = originAimedDepartureDate[0]
    departureMonth = originAimedDepartureDate[1]
    departureDay = originAimedDepartureDate[2]
    departureTime = originAimedDepartureTime.split("+")[0]
    return departureYear, departureDay, departureMonth, departureTime

def getBusesForTime(nowTime):
    headers = {'content-type': 'text/xml'}
    body = """<?xml version="1.0" ?>
    <S:Envelope xmlns:S="http://schemas.xmlsoap.org/soap/envelope/">
    <S:Body>
    <siriWS:GetStopMonitoringService xmlns:siriWS="http://new.webservice.namespace" xmlns="" xmlns:ns4="http://www.ifopt.org.uk/ifopt" xmlns:ns3="http://www.ifopt.org.uk/acsb" xmlns:siri="http://www.siri.org.uk/siri">
    <Request>
    <siri:RequestTimestamp>""" + nowTime + """+03:00</siri:RequestTimestamp>
    <siri:RequestorRef>AC309090</siri:RequestorRef>
    <siri:MessageIdentifier>AC:2019425:230410:981</siri:MessageIdentifier>
    <siri:StopMonitoringRequest version="2.7">
    <siri:RequestTimestamp>""" + nowTime + """+03:00</siri:RequestTimestamp>
    <siri:MessageIdentifier>0</siri:MessageIdentifier>
    <siri:PreviewInterval>PT30M</siri:PreviewInterval>
    <siri:StartTime>""" + nowTime + """+03:00</siri:StartTime>
    <siri:MonitoringRef>"""+BUS_STATION+"""</siri:MonitoringRef>
    <siri:MaximumStopVisits>100</siri:MaximumStopVisits>
    </siri:StopMonitoringRequest>
    </Request>
    </siriWS:GetStopMonitoringService>
    </S:Body>
    </S:Envelope>"""

    r = str(requests.post(WEB_SERVICE_URL, data=body, headers=headers).content)[2:-1].replace("\\'", "'")
    return r


while True:
    try:
        dateNowObj = datetime.datetime.now()+datetime.timedelta(hours=1)
        dateNow = str(dateNowObj).replace(" ", "T")

        response = getBusesForTime(dateNow)


        dateNowObj = datetime.datetime.now()

        tree = ET.fromstring(response)
        dataDictPerLine = {}

        for t in tree.iter():
            if "Answer" not in str(t):
                continue
            currentLineFields = {}

            for p in t.iter():
                if "MonitoredStopVisit" in str(p):
                    if len(currentLineFields) ==5:
                        dataDictPerLine[currentLineFields['line']] = currentLineFields
                    currentLineFields = {"latitude": -1, "longitude": -1}
                    continue

                if "PublishedLineName" in str(p):
                    currentLineFields["line"] = p.text
                    continue
                if "OriginAimedDepartureTime" in str(p):
                    p = p.text.replace("T", " ")
                    currentLineFields["originAimedDepartureTime"] = p
                    continue
                if "ExpectedArrivalTime" in str(p):
                    p = p.text.replace("T", " ")
                    currentLineFields["expectedArrivalTime"] = p
                    continue
                if "Longitude" in str(p):
                    currentLineFields["longitude"] = p.text
                    continue
                if "Latitude" in str(p):
                    currentLineFields["latitude"] = p.text
                    continue
                if "Description" in str(p):
                    print(p.text)


        conn = sqlite3.connect(DB_ROOT)

        for lineKey in dataDictPerLine:
            if len(lineKey) == 0:
                continue
            lineInfo = dataDictPerLine[lineKey]
            line = lineInfo["line"]
            longitude = lineInfo["longitude"]
            latitude = lineInfo["latitude"]

            originAimedDepartureTime = lineInfo["originAimedDepartureTime"]

            departureYear, departureDay, departureMonth, departureTime = formatDateTime(
                lineInfo["originAimedDepartureTime"])
            expectedArrivalYear, expectedArrivalDay, expectedArrivalMonth, expectedArrivalTime = formatDateTime(
                lineInfo["expectedArrivalTime"])

            datetime_object = dateNowObj + datetime.timedelta(seconds=60)
            datetime_object = str(datetime_object).replace(" ", "T")
            datetime_object = (datetime_object.replace("T", " ").split(".")[0]).split()[1]
            datetime_object = datetime_object.split(":")
            datetime_object[2] = '00'
            datetime_object = ":".join(datetime_object)

            tableLine = "'"+datetime_object+"',"+lineInfo["line"] + "," + departureYear + "," + departureDay + "," + departureMonth + ",'" + departureTime + "'," + expectedArrivalYear + "," + expectedArrivalDay + "," + expectedArrivalMonth + ",'" + expectedArrivalTime + "'," + str(latitude) + "," + str(longitude)

            query = "INSERT OR IGNORE INTO "+TABLE_NAME+" VALUES (" + tableLine + ")"



            if latitude == -1 and longitude == -1:
                print("skipping: "+str(query))
                continue

            if lineInfo["line"] == '610':
                threading.Thread(target=playSound, args=(1,)).start()



            print(query)
            conn.execute(query)
            conn.commit()

        conn.close()

        time.sleep(1)
    except:
        pass
