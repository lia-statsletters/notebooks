import os

def my_handler(event, context):
    try:
        print('Executing {} v.{}, weights {}MB in memory.'
              ''.format(os.environ['AWS_LAMBDA_FUNCTION_NAME'],
                        os.environ['AWS_LAMBDA_FUNCTION_VERSION'],
                        os.environ['AWS_LAMBDA_FUNCTION_MEMORY_SIZE']))
    except Exception as wtf:
        print(wtf.message)
