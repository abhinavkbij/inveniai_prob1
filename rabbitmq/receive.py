#!/usr/bin/env python
import pika, sys, os, json
import requests

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='sklearn')

    def callback(ch, method, properties, body):
        data = json.loads(body.decode('utf-8'))
        print(f" [x] Received {data}")

        # make api call for inference
        res = requests.get("http://localhost:8000/sklearn/infer", data=json.dumps({"data": json.loads(body.decode('utf-8'))}))
        print ("Response from api call: ", res.json())


    channel.basic_consume(queue='sklearn', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)