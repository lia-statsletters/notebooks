import boto3
import pprint

def read_queue(queue_url,sqs_client):
    """Barebones reader for everything in a queue."""
    response = sqs_client.receive_message(QueueUrl=queue_url)
    try:
        print ('{} messages retreived.'.format(len(response['Messages'])))
    except Exception as xx:
        print ('No messages in queue, or {}'.format(xx))
    return response

def write_to_queue(message_attrs, message_body,
                   queue_url, sqs_client, delay_secs=0):
        """Put a message in a queue."""
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            DelaySeconds=delay_secs,
            MessageAttributes=message_attrs,
            MessageBody=message_body
        )
        try:
            print('HTTPCode {} message successfully sent, '
                  '{} retries'.format(response['ResponseMetadata']['HTTPStatusCode'],
                                      response['ResponseMetadata']['RetryAttempts']))
        except Exception as xx:
            print('Problems sending the message: {}'.format(xx))

        return response


def main():
    playwithqueues()


def playwithqueues():
    queue_url = 'https://eu-west-1.queue.amazonaws.com/901610709743/boto_tut_sqs'
    sqs_client = boto3.client('sqs')
    pp = pprint.PrettyPrinter(indent=4)

    # Read the whole queue
    pp.pprint(read_queue(queue_url, sqs_client))

    # Make some message to send to the queue
    message_attrs = {'Title': {'DataType': 'String',
                               'StringValue': 'sometitle_tut'
                               }
                     }
    #message_body = ('some_message_body_tut')

    # Send the message to the queue
    # confirmation=write_to_queue(message_attrs, message_body,
    #               queue_url, sqs_client, delay_secs=0)
    # pp.pprint(confirmation)



if __name__ == "__main__":
    main()