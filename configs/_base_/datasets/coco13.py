dataset_info = dict(
    dataset_name='coco13',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='mid_head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_shoulder',
            id=1,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        2:
        dict(
            name='right_shoulder',
            id=2,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='left_elbow',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        4:
        dict(
            name='right_elbow',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        5:
        dict(
            name='left_wrist',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        6:
        dict(
            name='right_wrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        7:
        dict(
            name='left_hip',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        8:
        dict(
            name='right_hip',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        9:
        dict(
            name='left_knee',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        10:
        dict(
            name='right_knee',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        11:
        dict(
            name='left_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        12:
        dict(
            name='right_ankle',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={ # [ [11,9], [9,7], [7,8], [12,10], [10,8],// [8,2],[6,4],[4,2],[2,0],[7,1],//[5,3],[3,1],[1,0],[1,2] ]
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        3:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        4:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        5:
        dict(link=('right_hip', 'right_shoulder'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_wrist', 'right_elbow'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('right_elbow', 'right_shoulder'),id=7,color=[51, 153, 255]),
        8:
        dict(link=('right_shoulder', 'mid_head'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('left_hip', 'left_shoulder'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_wrist', 'left_elbow'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('left_elbow', 'left_shoulder'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_shoulder', 'mid_head'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('left_shoulder', 'right_shoulder'), id=13, color=[51, 153, 255]),
    },
    joint_weights=[
         1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
    ],
    sigmas=[
        0.09, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
