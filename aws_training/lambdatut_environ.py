from os import environ as envx

def handler(event, context):
    try:
        printable='lulz Executing {} v.{}, weights {} MB in memory.'.format(envx['AWS_LAMBDA_FUNCTION_NAME'],
                        envx['AWS_LAMBDA_FUNCTION_VERSION'],
                        envx['AWS_LAMBDA_FUNCTION_MEMORY_SIZE'])
    except Exception as wtf:
        print(wtf.message)
    print(printable)
    return printable


def main():
    handler(0,0)

if __name__ == "__main__":
    main()