# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/27 0:54
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mayavi import mlab



def numpy_plot_static_3D():
    class_map = np.array(
        [
            [[1, 1, 0],
             [1, 0, 1],
             [1, 1, 1]],
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]],
            [[1, 1, 1],
             [0, 0, 0],
             [0, 1, 0]]
        ]
    )

    x = []
    y = []
    z = []
    label = []
    class_num = 2
    for i in range(class_num):
        pos_x, pos_y, pos_z = np.nonzero(class_map == i)
        x.extend(list(pos_x))
        y.extend(list(pos_y))
        z.extend(list(pos_z))
        label.extend([i] * len(pos_x))

    # tooth_colors = mpl.colors.LinearSegmentedColormap.from_list(
    #     '牙齿种类颜色',
    #     ['#1f77b4', '#ff7f0e'],
    #     N=256
    # )

    tooth_colormap = mpl.colors.ListedColormap(['#1f77b4', '#ff7f0e'])
    plt.cm.register_cmap(name='tooth', cmap=tooth_colormap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=label, marker='o', cmap=plt.cm.get_cmap('tooth'), alpha=1, s=100)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()





def mayavi_plot_dynamic_3D():
    x = [[-1, 1, 1, -1, -1], [-1, 1, 1, -1, -1]]
    y = [[-1, -1, -1, -1, -1], [1, 1, 1, 1, 1]]
    z = [[1, 1, -1, -1, 1], [1, 1, -1, -1, 1]]

    s = mlab.mesh(x, y, z)
    mlab.show()

    def test_points3d():
        t = np.linspace(0, 4 * np.pi, 20)
        x = np.sin(2 * t)
        y = np.cos(t)
        z = np.cos(2 * t)
        s = 2 + np.sin(t)
        return mlab.points3d(x, y, z, s, colormap="Reds", scale_factor=.25)  # s (x,y,z)处标量的值 copper

    test_points3d()
    mlab.show()






if __name__ == '__main__':

    # numpy_plot_static_3D()

    # mayavi_plot_dynamic_3D()

    color_table = [
        [255, 255, 255, 0], # 0 background
        [255, 255, 255, 30], # 1 gum
        [255, 215, 0, 255], # 2 implant
        [85, 0, 0, 255], # 3 ul1
        [255, 0, 0, 255], # 4 ul2
        [85, 85, 0, 255], # 5 ul3
        [255, 85, 0, 255], # 6 ul4
        [85, 170, 0, 255], # 7, ul5
        [255, 170, 0, 255], # 8, ul6
        [85, 255, 0, 255], # 9 ul7
        [255, 255, 0, 255], # 10, ul8
        [0 ,0, 255, 255], # 11 ur1
        [170, 0, 255, 255], # 12 ur2
        [0, 85, 255, 255], # 13 ur3
        [170, 85, 255, 255], # 14 ur4
        [0, 170, 255, 255], # 15 ur5
        [170, 170, 255, 255], # 16 ur6
        [0, 255, 255, 255], # 17 ur7
        [170, 255, 255, 255], # 18 ur8
        [0, 0, 127, 255], # 19 bl1
        [170, 0, 127, 255], # 20 bl2
        [0, 85, 127, 255], # 21 bl3
        [170, 85, 127, 255], # 22 bl4
        [0, 170, 127, 255], # 23 bl5
        [170, 170, 127, 255], # 24 bl6
        [0, 255, 127, 255], # 25 bl7
        [170, 255, 127, 255], # 26 bl8
        [0, 0, 0, 255], # 27 br1
        [170, 0, 0, 255], # 28 br2
        [0, 85, 0, 255], # 29 br3
        [170, 85, 0, 255], # 30 br4
        [0, 170, 0, 255], # 31 br5
        [170, 170, 0, 255], # 32 br6
        [0, 255, 0, 255], # 33 br7
        [170, 255, 0, 255], # 34 br8
    ]









