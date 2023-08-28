#!/usr/bin/env python
import pika
import json
import pandas as pd

df = pd.read_csv("../../data.dat", delimiter="\t", header=None, names=["target", "text"])

message = [df["text"].tolist()[0]]

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='sklearn')

channel.basic_publish(exchange='', routing_key='sklearn', body=json.dumps(message))
print(f" [x] Sent {message}")
connection.close()