import tqdm
import glob
import os
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    print("only for test")

    savetxt = '/mnt/data4/yswangdata4/experiments/mm25evaluation/pcd_num.txt'

    if False:
        with open(savetxt, 'r') as f:
            recs = f.readlines()
        recs = [x.strip() for x in recs]
        with open('/mnt/data3/yswang2024_data3/dataset/blended_highres/process_scene', 'r') as f:
            validscenes = f.readlines()
        with open('/mnt/data3/yswang2024_data3/dataset/blended_highres/process_scene2', 'r') as f:
            validscenes = validscenes + f.readlines()
        validscenes = [x.strip() for x in validscenes]

        valid_rec = []
        for scene in validscenes:
            for rec in recs:
                if scene in rec:
                    valid_rec.append(rec)

        with open('/mnt/data4/yswangdata4/experiments/mm25evaluation/valid_pcd_num.txt', 'w') as f:
            f.writelines(valid_rec)



 #    blendedscene = ['/mnt/data4/yswangdata4/experiments/blended_helix_depfilt/helixdepfilt_*/point_cloud/iteration_30000/point_cloud.ply',
 # '/mnt/data4/yswangdata4/experiments/blended_helix_seg/helixseg_*/point_cloud/iteration_30000/point_cloud.ply',
 # '/mnt/data4/yswangdata4/experiments/blended_helix_segmulti/helixsegmultinomini_*/point_cloud/iteration_30000/point_cloud.ply',
 # '/mnt/data4/yswangdata4/experiments/blended_helix_segmulti/helixsegmulti_*/point_cloud/iteration_30000/point_cloud.ply',
 # ]
 #
 #    scenelist = []
 #    for rpth in blendedscene:
 #        scenes = glob.glob(rpth)
 #        scenelist = scenelist + scenes
 #
 #    def write_things(strings):
 #        with open(savetxt,'a') as f:
 #            f.write(str(strings))
 #
 #    for pcd_path in tqdm.tqdm(scenelist):
 #        pcd = o3d.io.read_point_cloud(pcd_path)
 #        npts = np.asarray(pcd.points).shape[0]
 #        write_things(pcd_path + ' ' + str(npts) + '\n')
 #        print(npts)

