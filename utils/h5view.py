import h5py

def print_attrs(name, obj):
    if isinstance(obj, h5py.Group):
         # node is a dataset
        print('-', name)
    else:
        print('----', obj)
    return

def h5view(f):
    '''
    output structure of a h5 file
    input: the address of h5 file
    '''

    try:
        file = h5py.File(f, 'r')
    except:
        print('cannot read the file')

    file.visititems(print_attrs)

    return


if __name__ == '__main__':
    h5view('HikariNiSequence.h5')