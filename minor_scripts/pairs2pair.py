import os
import argparse
import glob

def pairs2pair(input,output):
    with open(input, 'r') as f:
        data = f.readlines()
    n = len(data)
    with open(output, 'w') as f:
        f.write(n.__str__())
        f.write('\n')
        for i in range(n):
            line = data[i].strip().split(' ')
            line = [ele.split('.')[0] for ele in line]
            f.write(line[0])
            f.write('\n')
            f.write((len(line)-1).__str__())
            j = 1
            while j < len(line):
                f.write(' ')
                f.write(line[j])
                f.write(' ')
                f.write(str(5.24))
                j += 1
            f.write('\n')


if __name__ == '__main__':
    print("only for test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default="""/mnt/data3/yswang2024_data3/dataset/DTU_geoneus""")
    parser.add_argument('--output_root', type=str, default="""/mnt/data2/yswang2024/gaussian_surfels_data/2dgs/DTU""")
    args = parser.parse_args()

    inputs = sorted(glob.glob(os.path.join(args.input_root, 'scan*')))
    outputs = sorted(glob.glob(os.path.join(args.output_root, 'scan*')))

    for input,output in zip(inputs,outputs):
        pairs2pair(os.path.join(input,'pairs.txt'),os.path.join(output,'pair.txt'))
