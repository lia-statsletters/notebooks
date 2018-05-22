import os

def my_handler(event, context):
    foo=os.environ['foo']
    print('foo={}'.format(foo))
    return foo
