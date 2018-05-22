import pprint

#Other - CLI:
# - Create a bucket
# - Create a queue
# - Creating a lambda (telling aws that a package is a lambda, and its metadata)
#
# Other - Lambdas:
# - Reading from Environmental variables

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
    #playwithqueues()
    #playwithDynamo()

def playwithDynamo():

    from boto3 import resource as botoresource

    dynamodb = botoresource('dynamodb')
    create_populate_table(dynamodb)


def create_populate_table(dynamodb):
    # Instantiate a table resource object without actually
    # creating a DynamoDB table. Note that the attributes of this table
    # are lazy-loaded: a request is not made nor are the attribute
    # values populated until the attributes
    # on the table resource are accessed or its load() method is called.
    table = dynamodb.Table('users')

    # Print out some data about the table.
    # This will cause a request to be made to DynamoDB and its attribute
    # values will be set based on the response.
    print(table.creation_date_time)

    #Add item
    table.put_item(
        Item={
            'username': 'janedoe',
            'first_name': 'Jane',
            'last_name': 'Doe',
            'age': 25,
            'account_type': 'standard_user',
        }
    )

    #Get item
    response = table.get_item(Key={'username': 'janedoe',
                                   'last_name': 'Doe'})
    print(response['Item'])

    #Update item
    table.update_item(
        Key={'username': 'janedoe',
            'last_name': 'Doe'},
        UpdateExpression='SET age = :val1',
        ExpressionAttributeValues={
            ':val1': 26
        }
    )

    response = table.get_item(Key={'username': 'janedoe',
                              'last_name': 'Doe'} )
    print ("updated field: {}".format(response['Item']))

    #Delete item
    table.delete_item(Key={'username': 'janedoe',
                           'last_name': 'Doe'} )


def createTableDynamo(dynamodb,jsonfile='/home/lia/liaProjects/notebooks/dynamoJsonBotoTut.json'):

    from json import load

    try:
        with open(jsonfile) as f:
            schema = load(f)
    except Exception as badjsonfile:
        print('Problems loading the json')
        print(badjsonfile)

    # Create the DynamoDB table.
    table = dynamodb.create_table(
        TableName='users',
        KeySchema=schema["KeySchema"],
        AttributeDefinitions=schema["AttributeDefinitions"],
        ProvisionedThroughput=schema["ProvisionedThroughput"]
    )

    # Wait until the table exists.
    table.meta.client.get_waiter('table_exists').wait(TableName='users')

    # Print out some data about the table.
    print(table.item_count)


def playwithqueues():

    from boto3 import client as botoclient

    queue_url = 'https://eu-west-1.queue.amazonaws.com/901610709743/boto_tut_sqs'
    sqs_client = botoclient('sqs')
    pp = pprint.PrettyPrinter(indent=4)

    # Make some message to send to the queue
    message_attrs = {'Title': {'DataType': 'String',
                               'StringValue': 'sometitle_tut'
                               }
                     }
    message_body = ('some_message_body_tut')

    # Send the message to the queue
    confirmation=write_to_queue(message_attrs, message_body,
                  queue_url, sqs_client, delay_secs=0)
    pp.pprint(confirmation)

    # Read the whole queue
    pp.pprint(read_queue(queue_url, sqs_client))



if __name__ == "__main__":
    main()