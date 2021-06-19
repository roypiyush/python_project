import requests
from datetime import datetime


def callAPI():
    URL = "https://www.google.com"
    start = datetime.now()
    requests.get(url=URL)
    end = datetime.now()
    return (end - start).microseconds


if __name__ == '__main__':
    for i in range(200):
        delta = callAPI()
        print(delta)
