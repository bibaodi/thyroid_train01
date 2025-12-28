'''
convert multi label data to binary label format and combine with another binary format
If --binpath is provided, merge multi label data with binary data. Otherwise, convert multi label data.

version 2018/12/27 first version 
'''

import numpy as np
import os, argparse

class DataLoader:
    def __init__(self, datapath, dataset):
        self.X_train = np.load(f'{datapath}/imgs_train.npy')
        self.Y_train = np.load(f'{datapath}/imgs_mask_train.npy')
        self.ids_train = np.genfromtxt(f'{datapath}/imgs_fname_train.txt', dtype=np.unicode)

        self.X_test = np.load(f'{datapath}/imgs_test.npy')
        self.Y_test = np.load(f'{datapath}/imgs_mask_test.npy')
        self.ids_test = np.genfromtxt(f'{datapath}/imgs_fname_test.txt', dtype=np.unicode)

        if dataset in ['may', 'multi']:
            self.labels = np.loadtxt(f'{datapath}/labels.txt', dtype=np.unicode)
        else:
            self.labels = None    


def loadData(datapath, dataset):
    print("load data [{}] from {}".format(dataset, datapath))
    # get data
    dl = DataLoader(datapath, dataset)

    print("dl.X_train：", dl.X_train.shape)
    print("dl.Y_train", dl.Y_train.shape)
    print("dl.ids_train：", dl.ids_train.shape)
    # print("dl.ids_train：", dl.ids_train)
    print("dl.X_test:", dl.X_test.shape)
    print("dl.Y_test：", dl.Y_test.shape)
    print("dl.ids_test：", dl.ids_test.shape)
    # print("dl.ids_test：", dl.ids_test)

    print("dl.labels:", dl.labels)

    return dl

def combineDataSet(mul_path, bin_path, outdir, keylabel):
    mul_dl = loadData(mul_path, "multi")
    bin_dl = loadData(bin_path, "apr")

    keyindex = mul_dl.labels.tolist().index(keylabel)
    print(keyindex)

    print('Combine dataset. output to ' + outdir)
    os.makedirs(outdir, exist_ok=True)

    X_train = np.concatenate((mul_dl.X_train, bin_dl.X_train))
    print("X_train：", X_train.shape)
    np.save(f'{outdir}/imgs_train.npy', X_train)

    Y_train = np.concatenate((mul_dl.Y_train[...,keyindex:keyindex+1], bin_dl.Y_train))
    OutputYsize(Y_train, 'Y_train')
    np.save(f'{outdir}/imgs_mask_train.npy', Y_train)

    ids_train = np.concatenate((mul_dl.ids_train, bin_dl.ids_train))
    print("ids_train：", ids_train.shape)
    np.savetxt(f'{outdir}/imgs_fname_train.txt', ids_train, fmt='%s')

    
    X_test = np.concatenate((mul_dl.X_test, bin_dl.X_test))
    print("X_test：", X_test.shape)
    np.save(f'{outdir}/imgs_test.npy', X_test)

    Y_test = np.concatenate((mul_dl.Y_test[...,keyindex:keyindex+1], bin_dl.Y_test))
    OutputYsize(Y_test, 'Y_test')
    np.save(f'{outdir}/imgs_mask_test.npy', Y_test)

    ids_test = np.concatenate((mul_dl.ids_test, bin_dl.ids_test))
    print("ids_test：", ids_test.shape)
    np.savetxt(f'{outdir}/imgs_fname_test.txt', ids_test, fmt='%s')
    
    np.savetxt(f'{outdir}/labels.txt', mul_dl.labels[keyindex:keyindex+1], fmt='%s')

def OutputYsize(Y, name):
    print("{}：{}".format(name, Y.shape))
    y_labels = np.array(np.sum(Y, axis=(1, 2, 3)) > 0, dtype=np.uint8)
    positive_y_cnt = np.count_nonzero(y_labels)
    print('Postive sample in {} : {}'.format(name, positive_y_cnt))

def convertDataSet(mul_path, outdir, keylabel):
    mul_dl = loadData(mul_path, "multi")
    keyindex = mul_dl.labels.tolist().index(keylabel)
    print(keyindex)

    print('Convert dataset. output to ' + outdir)
    os.makedirs(outdir, exist_ok=True)

    print("X_train：", mul_dl.X_train.shape)
    np.save(f'{outdir}/imgs_train.npy', mul_dl.X_train)

    Y_train = mul_dl.Y_train[...,keyindex:keyindex+1]
    np.save(f'{outdir}/imgs_mask_train.npy', Y_train)
    OutputYsize(Y_train, 'Y_train')

    print("ids_train：", mul_dl.ids_train.shape)
    np.savetxt(f'{outdir}/imgs_fname_train.txt', mul_dl.ids_train, fmt='%s')

    print("X_test：", mul_dl.X_test.shape)
    np.save(f'{outdir}/imgs_test.npy', mul_dl.X_test)

    Y_test = mul_dl.Y_test[...,keyindex:keyindex+1]
    OutputYsize(Y_test, 'Y_test')
    np.save(f'{outdir}/imgs_mask_test.npy', Y_test)

    print("ids_test：", mul_dl.ids_test.shape)
    np.savetxt(f'{outdir}/imgs_fname_test.txt', mul_dl.ids_test, fmt='%s')
    
    np.savetxt(f'{outdir}/labels.txt', mul_dl.labels[keyindex:keyindex+1], fmt='%s')
 
def main():

    parser = argparse.ArgumentParser(description='Convert multi format to binary and combine with existing')

    parser.add_argument('--mulpath', dest='mul_path', type=str, required=True,
                        help='Path to folder where multi label format numpy data is stored.')
    parser.add_argument('--binpath', dest='bin_path', type=str, default='',
                        help='Path to folder where existing binary label format numpy data is stored. Empty means only convert multi format.')
    parser.add_argument('--outdir', dest='outdir', type=str, required=True,
                        help='Path to store the result')
    parser.add_argument('--keylabel', dest='keylabel', type=str, required=True, 
                        help='key label used for Y values')

    args = parser.parse_args()

    print(args)

    if args.bin_path:
        combineDataSet(args.mul_path, args.bin_path, args.outdir, args.keylabel)
    else:
        convertDataSet(args.mul_path, args.outdir, args.keylabel)  

if __name__ == '__main__':
    main() 
