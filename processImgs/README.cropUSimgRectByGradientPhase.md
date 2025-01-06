# readme cropUSimgRectByGradientPhase
- eton@241219 add python version for several issues.

## Issues
1. TypeError: Can't convert object to 'str' for 'filename'.;

2. logging.basicConfig(filename=log_file_name, encoding='utf-8', level=logging.DEBUG)
ValueError: Unrecognised argument(s): encoding ;

3. logger.WARNING(f"WARNING: pixel not grayscale , cannot remove color pixels.!!!")
AttributeError: 'Logger' object has no attribute 'WARNING' ;

### Reason
1. the develop python is 'Python 3.10.12' but daniel's environment is python3.6.2;
2. python3.10 support pathlib.Path implicit convert to str, but python3.6 not support;
3. logging module not support encoding parameter in py3.6 but supported in py3.10;
4. logging level function is `warning` not `WARNING`;

//


