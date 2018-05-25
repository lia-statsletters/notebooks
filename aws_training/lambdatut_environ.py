import os

def handler(event, context):
    try:
        print('Executing {} v.{}, weights {} MB in memory.'
              ''.format(os.environ['AWS_LAMBDA_FUNCTION_NAME'],
                        os.environ['AWS_LAMBDA_FUNCTION_VERSION'],
                        os.environ['AWS_LAMBDA_FUNCTION_MEMORY_SIZE']))
    except Exception as wtf:
        print(wtf.message)


def main():
    handler(0,0)

if __name__ == "__main__":
    main()