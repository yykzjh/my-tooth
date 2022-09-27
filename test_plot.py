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

    numpy_plot_static_3D()

    mayavi_plot_dynamic_3D()









