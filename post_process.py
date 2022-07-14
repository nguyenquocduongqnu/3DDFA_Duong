import open3d as o3d
import numpy as np
import copy
import math
import time

head = o3d.io.read_triangle_mesh('configs/ori_face_small.obj')
head = head.simplify_quadric_decimation(30000)

long_black_hair = o3d.io.read_triangle_mesh(f'configs/only_hair_long_black.obj')
short_black_hair = o3d.io.read_triangle_mesh(f'configs/only_hair_short_black.obj')
blonde_black_hair = o3d.io.read_triangle_mesh(f'configs/only_hair_blonde_black.obj')

long_hair_indi = [2.9855026, 53.19723651, 86.09840478]
short_hair_indi = [1.07962544, 22.08013648, 67.93180361]
blonde_hair_indi = [-6.34703412, 43.02308654, 91.02896217]

hairstyles = [
    {
        'name': 'long_black',
        'indi': long_hair_indi,
        'obj': long_black_hair
    },
    {
        'name': 'short_black',
        'indi': short_hair_indi,
        'obj': short_black_hair
    },
    {
        'name': 'blonde_black',
        'indi': blonde_hair_indi,
        'obj': blonde_black_hair
    },
]


def distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def rgba2rgb( rgba ):
    rgba = rgba/255.0
    rgb = np.zeros( (len(rgba), 3), dtype='float32' )
    rgb[:, 0] = (1.0 - rgba[:, 3]) * rgb[:, 0] + (rgba[:, 3] * rgba[:, 0])
    rgb[:, 1] = (1.0 - rgba[:, 3]) * rgb[:, 1] + (rgba[:, 3] * rgba[:, 1])
    rgb[:, 2] = (1.0 - rgba[:, 3]) * rgb[:, 2] + (rgba[:, 3] * rgba[:, 2])

    return rgb

def post_process(vertices, colors, faces,  hair_style='long_black'):
    face = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                           triangles=o3d.utility.Vector3iVector(faces))
    face.vertex_colors = o3d.utility.Vector3dVector(colors)
    face_height = max(np.array(face.vertices)[:, 1]) - min(np.array(face.vertices)[:, 1])
    head_height = max(np.array(head.vertices)[:, 1]) - min(np.array(head.vertices)[:, 1])

    face.vertices = o3d.utility.Vector3dVector(np.array(face.vertices) * (180 / face_height))
    face_height = max(np.array(face.vertices)[:, 1]) - min(np.array(face.vertices)[:, 1])
    countlopp = 0
    max_z_nose = np.argmax(np.array(face.vertices)[:, 2])  # tìm tọa độ của đỉnh mũi
    while 1:
        countlopp += 1
        list_idx_nn = []  # những điểm gần mũi nhất theo trục x
        vertex_nose = face.vertices[max_z_nose]
        #     print(vertex_nose)
        for idx in range(len(face.vertices)):
            if vertex_nose[0] + 2 > face.vertices[idx][0] > vertex_nose[
                0] - 2:  # những điểm gần theo trục x 4 bốn vị với điểm mũi
                list_idx_nn.append(face.vertices[idx])  # tọa độ chứ ko lấy index

        list_idx_nn_sorted = sorted(list_idx_nn, key=lambda x: x[1])  # sắp xếp theo tăng dần của y
        #     print(list_idx_nn_sorted[0]) # đây là đường thẳng đứng hiện tại.

        # cần tìm điểm xa MŨI nhất mà nằm ở phần dưới của mũi

        list_idx_ln = []  # danh sách những điểm nằm dưới mũi
        for idx in range(len(face.vertices)):
            if face.vertices[idx][1] < vertex_nose[1]:  # những điểm có y nhỏ hơn y của mũi
                # tính khoảng cách tới mũi luôn, tính theo x y thôi, ko tính z
                p1 = [vertex_nose[0], vertex_nose[1], 0]
                p2 = [face.vertices[idx][0], face.vertices[idx][1], 0]
                dis_to_nose = distance_3d(p1, p2)
                # đưa vài list
                list_idx_ln.append([face.vertices[idx], dis_to_nose])
        # sắp xếp danh sách theo khoảng cách
        list_idx_ln_sorted = sorted(list_idx_ln, key=lambda x: x[1])

        #     print(list_idx_ln_sorted[-1]) # điểm xa nhất
        x_ln = list_idx_ln_sorted[-1][0][0]
        x_nn = list_idx_nn_sorted[0][0]
        if x_ln == x_nn:  # nếu điểm xa nhất cũng  là điểm thấp nhất thì chỉnh mặt đã xong
            break
        else:
            if x_ln > vertex_nose[0]:  # điểm xa nhất ở lệch bên phải thì xoay âm
                ratio_rotate = -0.1 / 18  # 1 độ trên 180 độ
            else:
                ratio_rotate = 0.1 / 18
            face_matrix = face.get_rotation_matrix_from_axis_angle([0, 0, ratio_rotate])
            face.rotate(face_matrix)
        if countlopp > 20: break

    scale_ratio = face_height * 1.4 / head_height
    head.vertices = o3d.utility.Vector3dVector(np.array(head.vertices) * scale_ratio)
    top_head_nose_id = np.argmax(np.array(head.vertices)[:, 2])
    top_face_nose_id = np.argmax(np.array(face.vertices)[:, 2])
    align_range = head.vertices[top_head_nose_id] - face.vertices[top_face_nose_id]
    face.vertices = o3d.utility.Vector3dVector(np.array(face.vertices) + align_range)

    """rotate head"""
    rotate_head = copy.deepcopy(head)
    rotation_matrix = rotate_head.get_rotation_matrix_from_axis_angle([0.5 / 18, 0, 0])
    rotate_head.rotate(rotation_matrix)

    """remove face border"""
    max_z_face = max(np.array(face.vertices)[:, 2])
    min_z_face = min(np.array(face.vertices)[:, 2])
    max_y_face = max(np.array(face.vertices)[:, 1])
    min_y_face = min(np.array(face.vertices)[:, 1])
    y_remove = max_y_face - (max_y_face - min_y_face) / 18
    border_mask = (np.array(face.vertices)[:, 2] < (((max_z_face - min_z_face) * 0.35) + min_z_face)) | (
                np.array(face.vertices)[:, 1] > y_remove)
    face.remove_vertices_by_mask(border_mask)

    countlopp = 0
    while 1:
        countlopp += 1
        max_z_face = max(np.array(face.vertices)[:, 2])
        min_z_face = min(np.array(face.vertices)[:, 2])
        max_y_face = max(np.array(face.vertices)[:, 1])
        min_y_face = min(np.array(face.vertices)[:, 1])

        # find nose point of face and head
        nose_face_vertice = np.argmax(np.array(face.vertices)[:, 2])
        nose_head_vertice = np.argmax(np.array(rotate_head.vertices)[:, 2])
        nose_point_face = face.vertices[nose_face_vertice]
        nose_point_head = rotate_head.vertices[nose_head_vertice]
        # distance between 2 nose points
        dis_move = nose_point_head - nose_point_face
        # moving head at the same nose point
        for idx, vertice in enumerate(rotate_head.vertices):
            rotate_head.vertices[idx] -= dis_move

        # điểm cao nhất của mũi trên mặt
        face_idx = []
        for idx in range(len(rotate_head.vertices)):
            if (rotate_head.vertex_colors[idx][0] != 0.0) & (rotate_head.vertices[idx][2] > min_z_face) & (
                    rotate_head.vertices[idx][1] < max_y_face) & (rotate_head.vertices[idx][1] > (min_y_face - 50)):
                face_idx.append(idx)

        face_of_head = rotate_head.select_by_index(face_idx)
        nose_face_vertice = np.argmax(np.array(face.vertices)[:, 2])

        # lấy điểm cao nhất giữa mũi
        list_idx_nn = []  # những điểm gần mũi nhất theo trục x
        vertex_nose = face.vertices[nose_face_vertice]
        #     print(vertex_nose)
        for idx in range(len(face.vertices)):
            if vertex_nose[0] + 2 > face.vertices[idx][0] > vertex_nose[
                0] - 2:  # những điểm gần theo trục x 4 bốn vị với điểm mũi
                list_idx_nn.append(face.vertices[idx])  # tọa độ chứ ko lấy index

        list_idx_nn_sorted = sorted(list_idx_nn, key=lambda x: x[1])  # sắp xếp theo tăng dần của y
        #     print(list_idx_nn_sorted[-1]) # đây là đường thẳng đứng hiện tại.
        # có được tọa độ z của điểm cao nhất theo trục mũi
        z_max_face = list_idx_nn_sorted[-1][2]
        #     print(z_max_face)

        nose_face_vertice = np.argmax(np.array(face_of_head.vertices)[:, 2])

        # lấy điểm cao nhất giữa mũi
        list_idx_nn_foh = []  # những điểm gần mũi nhất theo trục x
        vertex_nose = face_of_head.vertices[nose_face_vertice]
        #     print(vertex_nose)
        for idx in range(len(face_of_head.vertices)):
            if vertex_nose[0] + 2 > face_of_head.vertices[idx][0] > vertex_nose[
                0] - 2:  # những điểm gần theo trục x 4 bốn vị với điểm mũi
                list_idx_nn_foh.append(face_of_head.vertices[idx])  # tọa độ chứ ko lấy index

        list_idx_nn_foh_sorted = sorted(list_idx_nn_foh, key=lambda x: x[1])  # sắp xếp theo tăng dần của y
        #     print(list_idx_nn_foh_sorted[-1]) # đây là đường thẳng đứng hiện tại.
        # có được tọa độ z của điểm cao nhất theo trục mũi
        z_max_face_foh = list_idx_nn_foh_sorted[-1][2]
        #     print(z_max_face_foh)
        if (z_max_face_foh - 0.5) < z_max_face < (z_max_face_foh + 0.5):
            break
        else:
            if z_max_face_foh > z_max_face:
                rotate_ratio = -0.1
            else:
                rotate_ratio = 0.1
        rotation_matrix = rotate_head.get_rotation_matrix_from_axis_angle([rotate_ratio / 18, 0, 0])
        rotate_head.rotate(rotation_matrix)
        if countlopp > 20: break

    face = face.subdivide_midpoint(1)
    max_z = max(np.array(rotate_head.vertices)[:, 2])
    min_z = min(np.array(rotate_head.vertices)[:, 2])

    depth_size = (max_z - min_z) / 3
    remove_depth = max_z - depth_size
    width_of_foh = max(np.array(face_of_head.vertices)[:, 0]) - min(np.array(face_of_head.vertices)[:, 0])
    width_of_face = max(np.array(face.vertices)[:, 0]) - min(np.array(face.vertices)[:, 0])

    """scale face by width ratio"""
    scale_ratio_width = width_of_foh / width_of_face
    # scale_ratio_height = height_of_face / height_of_foh
    # print(scale_ratio_width, scale_ratio_height)
    for idx in range(len(face.vertices)):
        face.vertices[idx][0] *= scale_ratio_width
        face.vertices[idx][1] *= scale_ratio_width
    #     head.vertices[idx][1] *= scale_ratio_height

    height_of_foh = max(np.array(face_of_head.vertices)[:, 1]) - min(np.array(face_of_head.vertices)[:, 1])
    height_of_face = max(np.array(face.vertices)[:, 1]) - min(np.array(face.vertices)[:, 1])

    tmp_ratio = height_of_face / height_of_foh
    scale_ratio_height = height_of_foh / height_of_face
    for idx in range(len(face.vertices)):
        face.vertices[idx][1] *= scale_ratio_height

    """align center(x,y) 2 faces"""
    height_of_foh = max(np.array(face_of_head.vertices)[:, 1])
    height_of_face = max(np.array(face.vertices)[:, 1])
    dis_height = height_of_face - height_of_foh
    width_of_foh = max(np.array(face_of_head.vertices)[:, 0])
    width_of_face = max(np.array(face.vertices)[:, 0])
    dis_width = width_of_face - width_of_foh
    for idx in range(len(face.vertices)):
        face.vertices[idx][0] -= dis_width
        face.vertices[idx][1] -= dis_height
    #   face.vertices[idx][2] += 10

    rotate_head = rotate_head.subdivide_midpoint(1)

    # làm lại face of head với diện tích to hơn smaller face
    face_idx_again = []
    min_z_face = min(np.array(face.vertices)[:, 2])
    max_y_face = max(np.array(face.vertices)[:, 1])
    for idx in range(len(rotate_head.vertices)):
        if (rotate_head.vertex_colors[idx][0] != 0.0) & (rotate_head.vertices[idx][2] > min_z_face + 5) & (
                rotate_head.vertices[idx][1] < max_y_face - 5) & (rotate_head.vertices[idx][1] > (min_y_face - 50)):
            face_idx_again.append(idx)
    face_of_head = rotate_head.select_by_index(face_idx_again)

    face_removed_head = copy.deepcopy(rotate_head)
    face_removed_head.remove_vertices_by_index(face_idx_again)
    face_full = face_removed_head + face
    face_full = face_full.filter_smooth_laplacian()
    face_full = face_full.filter_smooth_taubin()
    face_full = face_full.filter_smooth_simple()

    min_head_bound = rotate_head.get_min_bound()

    list_head = []
    for hairstyle in hairstyles:
        dis_ftf = face.get_center() - hairstyle['indi']
        hair = hairstyle['obj']

        for idx_hair in range(len(hair.vertices)):
            hair.vertices[idx_hair] += dis_ftf

        list_new_hair = []
        max_hair_bound = hair.get_max_bound()
        min_hair_bound = hair.get_min_bound()

        for idx_hair in range(len(hair.vertices)):
            if hair.vertices[idx_hair][1] > min_head_bound[1]:
                list_new_hair.append(idx_hair)
        hair = hair.select_by_index(list_new_hair)

        max_hair_bound = hair.get_max_bound()
        min_hair_bound = hair.get_min_bound()
        for idx_hair in range(len(hair.vertices)):
            if hair.vertices[idx_hair][2] < ((max_hair_bound[2] + min_hair_bound[2]) / 2):
                hair.vertices[idx_hair][2] -= 20

        max_hair_bound = hair.get_max_bound()
        min_hair_bound = hair.get_min_bound()
        for idx_hair in range(len(hair.vertices)):
            if hair.vertices[idx_hair][1] > ((max_hair_bound[1] + min_hair_bound[1]) / 2):
                hair.vertices[idx_hair][1] += 15

        black_hair_head = face_full + hair
        list_head.append({
            'obj_name' : hairstyle['name'],
            'obj': black_hair_head.simplify_quadric_decimation(100000)
        })
        
        hair.paint_uniform_color([1.0, 0.88235294, 0.38431373])
        yellow_hair_head = face_full + hair
        list_head.append({
            'obj_name' : hairstyle['name'].replace('black','yellow'),
            'obj': yellow_hair_head.simplify_quadric_decimation(100000)
        })
    return list_head

def export(mesh_lst, obj_path, img_path):
    for mesh in mesh_lst:
        mesh['obj_path'] = obj_path.replace('.obj',mesh['obj_name']+'.obj')
        mesh['img_path'] = img_path.replace('.png',mesh['obj_name']+'.png')
    for mesh in mesh_lst:
        nan_ids = np.argwhere(np.isnan(np.sum(mesh['obj'].vertices, axis=1))).flatten()
        if len(nan_ids) > 0:
            for idx in nan_ids:
                mesh['obj'].vertices[idx] = mesh['obj'].vertices[idx - 1]
                mesh['obj'].vertex_colors[idx] = mesh['obj'].vertex_colors[idx -1]
        o3d.io.write_triangle_mesh(mesh['obj_path'],mesh['obj'],print_progress=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh['obj'])
        vis.update_geometry(mesh['obj'])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(mesh['img_path'])
        vis.destroy_window()
    return mesh_lst
