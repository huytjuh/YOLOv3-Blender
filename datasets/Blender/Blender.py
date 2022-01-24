import os
import bpy
import sys

import pandas as pd
import numpy as np

import glob
from mathutils import Matrix
from math import radians
import cv2

path = '//code1/storage/2014-0353_generaleye_ux/Huy/Data'
os.chdir(path)

# CLEAR ENVIRONMENT
def clear_env():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    return
        
def mega_purge():
    orphan_ob = [o for o in bpy.data.objects if not o.users]
    while orphan_ob:
        bpy.data.objects.remove(orphan_ob.pop())
        
    orphan_mesh = [m for m in bpy.data.meshes if not m.users]
    while orphan_mesh:
        bpy.data.meshes.remove(orphan_mesh.pop())
        
    orphan_mat = [m for m in bpy.data.materials if not m.users]
    while orphan_mat:
        bpy.data.materials.remove(orphan_mat.pop())

    def purge_node_groups():   
        orphan_node_group = [g for g in bpy.data.node_groups if not g.users]
        while orphan_node_group:
            bpy.data.node_groups.remove(orphan_node_group.pop())
        if [g for g in bpy.data.node_groups if not g.users]: purge_node_groups()
    purge_node_groups()
        
    orphan_texture = [t for t in bpy.data.textures if not t.users]
    while orphan_texture:
        bpy.data.textures.remove(orphan_texture.pop())

    orphan_images = [i for i in bpy.data.images if not i.users]
    while orphan_images:
        bpy.data.images.remove(orphan_images.pop())

    orphan_cameras = [c for c in bpy.data.cameras if not c.users]
    while orphan_cameras:
        bpy.data.cameras.remove(orphan_cameras.pop())
        
    orphan_lights = [l for l in bpy.data.lights if not l.users]
    while orphan_lights:
        bpy.data.lights.remove(orphan_lights.pop())
        
    orphan_light_probes = [lp for lp in bpy.data.lightprobes if not lp.users]
    while orphan_light_probes:
        bpy.data.lightprobes.remove(orphan_light_probes.pop())
        
    orphan_scenes = [s for s in bpy.data.scenes if not s.users]
    while orphan_scenes:
        bpy.data.scenes.remove(orphan_scenes.pop())
             

#########################################
### LOAD OBJECTS & CREATE ENVIRONMENT ###
#########################################

def load_environment():
    scn = bpy.context.scene.collection.children[0]

    # LOAD ENVIRONMENT
    list_room = glob.glob('hospital recovery room 3D model files/*.fbx')
    room = list_room[1]
    bpy.ops.import_scene.fbx(filepath=room)
    bpy.ops.file.find_missing_files(directory='hospital recovery room 3D model files/uploads_files_228128_RecoveryRoomTextures')
    for scene in bpy.data.objects:
        scene.location = (0,0,0)
    return

def load_lights(list_lamp, list_lamp_main, list_sun):
    scn = bpy.context.scene.collection.children[0]
    
    # ADD LAMP
    for lamp_name in list(list_lamp):
        lamp_params = list_lamp[lamp_name]
        
        lamp = bpy.data.lights.new(name=lamp_name, type='SPOT')
        lamp.energy = lamp_params['energy']
        lamp.specular_factor = lamp_params['specular_factor']
        lamp.shadow_soft_size = lamp_params['shadow_soft_size']
        lamp.spot_size = lamp_params['spot_size']
        lamp.spot_blend = lamp_params['spot_blend']
        lamp_obj = bpy.data.objects.new(name=lamp_name, object_data=lamp)
        lamp_obj.location = lamp_params['location']
        lamp_obj.rotation_euler = lamp_params['rotation_euler']
        scn.objects.link(lamp_obj)

    # ADD MAIN LAMP
    lamp_main = bpy.data.lights.new(name='Lamp_main', type='SPOT')
    lamp_main.energy = list_lamp_main['energy']
    lamp_main.specular_factor = list_lamp_main['specular_factor']
    lamp_main.shadow_soft_size = list_lamp_main['shadow_soft_size']
    lamp_main.spot_size = list_lamp_main['spot_size']
    lamp_main.spot_blend = list_lamp_main['spot_blend']
    lamp_main_obj = bpy.data.objects.new(name='Lamp_main', object_data=lamp_main)
    lamp_main_obj.location = list_lamp_main['location']
    lamp_main_obj.rotation_euler = list_lamp_main['rotation_euler']
    scn.objects.link(lamp_main_obj)

    # ADD SUN
    sun = bpy.data.lights.new(name="Sun", type='AREA')
    sun.color = list_sun['color']
    sun.energy = list_sun['energy']
    sun.specular_factor = list_sun['specular_factor']
    sun.shape = 'DISK'
    sun.size = list_sun['size']
    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun)
    sun_obj.location = list_sun['location']
    sun_obj.rotation_euler = list_sun['rotation_euler']
    scn.objects.link(sun_obj)

    # ADD REFLECTION CUBEMAP
    refl = bpy.data.lightprobes.new(name='ReflectionCubeMap', type='CUBE')
    refl.intensity = 3
    refl.clip_start = 35
    refl_obj = bpy.data.objects.new(name='ReflectionCubeMap', object_data=refl)
    scn.objects.link(refl_obj)
    return

def load_characters(file_path, location, rotation):
    scn = bpy.context.scene.collection.children[0]
    bpy.ops.import_scene.fbx(filepath=file_path)
    
    location = [float(x.strip()) for x in location[1:-1].split(',')]
    rotation = [radians(float(x.strip())) for x in rotation[1:-1].split(',')]
    bpy.data.objects['Armature'].location = location
    bpy.data.objects['Armature'].rotation_euler = rotation
    return


############################
### CREATE CAMERA ANGLES ###
############################

def load_viewpoints(list_cam):
    scn = bpy.context.scene.collection.children[0]
    
    # INITIALIZE VIEWPOINTS
    FRONT = list_cam['S01']
    FRONTLEFT = list_cam['S02']
    FRONTRIGHT = list_cam['S03']
    TOP = list_cam['S04']

    # CREATE THE FOUR DIFFERENT VIEWPOINTS
    view_points = [['S01', FRONT], ['S02', FRONTRIGHT], ['S03', FRONTLEFT], ['S04', TOP]]
    for angle in view_points:
        name = 'Camera_' + angle[0]
        coord = angle[1]
            
        # CREATE CAMERA OBJECT
        camera = bpy.data.cameras.new(name=name)
        camera.lens = coord['lens']
        camera.shift_x = coord['shift_x']
        camera.shift_y = coord['shift_y']
        
        # CAMERA SETTINGS
        camera_obj = bpy.data.objects.new(name, camera)
        camera_obj.location = coord['location']
        camera_obj.rotation_euler = coord['rotation_euler']
        
        scn.objects.link(camera_obj)
    return


###########################
### CREATE TRAINING SET ###
###########################

def export_IMG(out_name, out_dir, end_frame):
 
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 20
    scene.cycles.device = 'GPU'
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'

    # INITIALIZE SCENE SETTINGS
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.image_settings.file_format = 'PNG'

    scene.view_layers["View Layer"].use_pass_object_index = True
    scene.use_nodes = True

    # IMAGE SEGMENTATION
    body_parts = ['Body', 'Bottoms', 'default', 'Eyelashes', 'Hair', 'Tops']
    index = 100
    for part in body_parts:
        obj = bpy.data.objects[part]
        obj.pass_index = index

    tree = scene.node_tree

    # EMPTY NODES
    for node in tree.nodes:
        tree.nodes.remove(node)

    # RENDER LAYERES NODE
    render_node = tree.nodes.new(type='CompositorNodeRLayers')
    render_node.location = 0,500

    # COMPOSITE NODE
    comp_node = tree.nodes.new(type='CompositorNodeComposite')
    comp_node.location = 500, 500
    tree.links.new(render_node.outputs[0], comp_node.inputs[0])

    # ID_MASK CONVERTER NODE
    mask_node = tree.nodes.new(type='CompositorNodeIDMask')
    mask_node.index = index
    tree.links.new(render_node.outputs[3], mask_node.inputs[0])
    mask_node.location = 300, 300

    # VIEWER NODE
    viewer_node = tree.nodes.new(type='CompositorNodeViewer')
    viewer_node.location = 500, 100
    tree.links.new(mask_node.outputs[0], viewer_node.inputs[0])
    
    # OUTPUT NODE
    output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    output_node.location = 500, 300
    tree.links.new(mask_node.outputs[0], output_node.inputs[0])
    bpy.data.scenes['Scene'].node_tree.nodes['File Output'].format.color_mode = 'BW'
    bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = out_dir['base']
    
    # NORMALIZE NODE
    output_node.layer_slots.new('Depth_map')
    normalize_node = tree.nodes.new(type='CompositorNodeNormalize')
    normalize_node.location = 300, 100
    tree.links.new(render_node.outputs[2], normalize_node.inputs[0])
    
    # COLORRAMP
    colorramp_node = tree.nodes.new(type='CompositorNodeValToRGB')
    colorramp_node.location = 300, -50
    colorramp_node.color_ramp.elements.new(0.8)
    tree.links.new(normalize_node.outputs[0], colorramp_node.inputs[0])
    tree.links.new(colorramp_node.outputs[0], output_node.inputs[1])
    
    # CREATE BOUNDING BOX
    width  = int(scene.render.resolution_x*scene.render.resolution_percentage/100)
    height = int(scene.render.resolution_y*scene.render.resolution_percentage/100)
    depth  = 4

    list_camera = ['S01', 'S02', 'S03', 'S04']
    for view in list_camera:
        for i in range(1, int(end_frame)+1, 10):
            bpy.context.scene.frame_set(i)
            bpy.context.scene.camera = bpy.data.objects['Camera_' + view]
            
            output_node.file_slots[0].path = 'Mask/' + out_dir['file'] + '/' + out_name + view + '_###'
            output_node.file_slots[1].path = 'Depth/' + out_dir['file'] + '/' + out_name + view + '_###'
               
            file_path = out_dir['base'] + 'RGB/' + out_dir['file'] + '/' + out_name + view + '_{:03d}'.format(i)
            scene.render.filepath = file_path
            bpy.ops.render.render(write_still=1)
            
            pixels = np.array(bpy.data.images['Viewer Node'].pixels[:]).reshape([height, width, depth])
            pixels = np.array([[pixel[0] for pixel in row] for row in pixels])
            
            bbox = np.argwhere(pixels)
            if len(bbox) == 0:
                continue
                
            y_min, x_min, y_max, x_max = *bbox.min(0), *bbox.max(0) + 1

            with open('{}.txt'.format(path + file_path.replace('//', '/')), 'w') as out_file:
                out_file.write('{}.png'.format(file_path))
                box_str = '{:.0f},{:.0f},{:.0f},{:.0f},{}'.format(x_min, height-y_max, x_max, height-y_min, 0)
                out_file.write(' ' + box_str)


#############################
### CREATE CLASS NAME TXT ###
#############################

def create_class_names():
    df_class_names = pd.Series('Human')
    df_class_names.to_csv('class_names.txt', header=False)


#####################
### MAIN FUNCTION ###
#####################

def initialize_parameters():
    
    # INITIALIZE VIEWPOINTS
    list_cam = {}
    list_cam['S01'] = {'lens': 20, 'shift_x': 0.05, 'shift_y': 0.07,
                         'location': (0,-2.5,2), 'rotation_euler': (radians(65), radians(0), radians(360))}
    list_cam['S02'] = {'lens': 22, 'shift_x': 0.03, 'shift_y': 0.04, 
                             'location': (-2,-3,2.2), 'rotation_euler': (radians(65), radians(0), radians(320))}
    list_cam['S03'] = {'lens': 22, 'shift_x': 0.03, 'shift_y': 0.04, 
                             'location': (1.8,-2.3,2), 'rotation_euler': (radians(65), radians(0), radians(400))}
    list_cam['S04'] = {'lens': 15, 'shift_x': 0.05, 'shift_y': 0.11,
                       'location': (0,1.6,2.2), 'rotation_euler': (radians(50), radians(0), radians(180))}
    
    # LAMP
    list_lamp = {}
    list_lamp['Lamp01'] = {'energy': 2, 'specular_factor': 20, 'shadow_soft_size': 0.5, 'spot_size': 160, 'spot_blend': 0.8,
                           'location': (-1.39,1.39,2.75), 'rotation_euler': (radians(0), radians(0), radians(101))}
    list_lamp['Lamp02'] = {'energy': 2, 'specular_factor': 20, 'shadow_soft_size': 0.5, 'spot_size': 160, 'spot_blend': 0.8,
                           'location': (1.98,0.97,2.30), 'rotation_euler': (radians(0), radians(0), radians(101))}
    list_lamp['Lamp03'] = {'energy': 10, 'specular_factor': 2, 'shadow_soft_size': 0.5, 'spot_size': 180, 'spot_blend': 0.35,
                           'location': (0.36,1.43,2.48), 'rotation_euler': (radians(-20), radians(0), radians(180))}
    
    # LAMP CEILING
    settings_lamp = {}
    settings_lamp['Low'] = {'energy': 10, 'specular_factor': 2, 'shadow_soft_size': 1, 'spot_size': 180, 'spot_blend': 0.45}
    settings_lamp['Mid'] = {'energy': 100, 'specular_factor': 2, 'shadow_soft_size': 1, 'spot_size': 180, 'spot_blend': 0.45}
    settings_lamp['High'] = {'energy': 250, 'specular_factor': 2, 'shadow_soft_size': 1, 'spot_size': 180, 'spot_blend': 0.45}
    
    list_lamp_main = {}
    for dens in list(settings_lamp):
        list_lamp_main[dens] = {'location': (0.59,-1.48,2.93), 'rotation_euler': (radians(-20), radians(0), radians(180))}
        list_lamp_main[dens].update(settings_lamp[dens])
    
    # SUN
    settings_sun = {}
    settings_sun['Low'] = {'color': (1,0.47,0.44), 'energy': 1, 'specular_factor': 1, 'size': 2}
    settings_sun['Mid'] = {'color': (1,0.47,0.44), 'energy': 50, 'specular_factor': 1, 'size': 2}
    settings_sun['High'] = {'color': (1,0.47,0.44), 'energy': 220, 'specular_factor': 1, 'size': 2}
    
    list_sun = {}
    for dens in list(settings_sun):
        list_sun[dens] = {'location': (2.24,0.29,2.3), 'rotation_euler': (radians(10), radians(-10), radians(-30))}
        list_sun[dens].update(settings_sun[dens])
    
    return list_cam, list_lamp, list_lamp_main, list_sun
        
        
if __name__ == "__main__":
    
    # LOAD LIST OF FILE NAMES
    df_map = pd.read_csv('file_names.csv')
    
    list_file = []
    for code1, map1 in zip(df_map.loc[df_map['Level'] == 1, 'Code'], df_map.loc[df_map['Level'] == 1, 'Map']):
        for code2, map2 in zip(df_map.loc[df_map['Level'] == 2, 'Code'], df_map.loc[df_map['Level'] == 2, 'Map']):
            for code3, map3 in zip(df_map.loc[df_map['Level'] == 3, 'Code'], df_map.loc[df_map['Level'] == 3, 'Map']):
                for code4, map4 in zip(df_map.loc[df_map['Level'] == 4, 'Code'], df_map.loc[df_map['Level'] == 4, 'Map']):
                    for code5, map5, file5, end_frame, location, rotation in zip(df_map.loc[df_map['Level'] == 5, 'Code'], df_map.loc[df_map['Level'] == 5, 'Map'], df_map.loc[df_map['Level'] == 5, 'File_name'],
                                                                                 df_map.loc[df_map['Level'] == 5, 'End_frame'], df_map.loc[df_map['Level'] == 5, 'Location'], df_map.loc[df_map['Level'] == 5, 'Rotation']): 
                        for code6, light in zip(df_map.loc[df_map['Level'] == 6, 'Code'], df_map.loc[df_map['Level'] == 6, 'File_name']):                    
                            file_name = code1 + code2 + code3 + code4 + '_' + code5 + '_' + code6 + '_'
                            file_path = 'Characters and Animations/' + map1 + '/' + map2 + '/' + map3 + '/' + map4 + '/' + map5 + '/' + file5 + '.fbx'
                            list_file.append({'file_name': file_name, 'file_path': file_path, 'light': light,
                                              'end_frame': end_frame, 'location': location, 'rotation': rotation})
   
   
    list_file_cluster = np.array_split(list_file, 9)
    for file in list_file_cluster[int(sys.argv[-1])-1]:        
    #for file in list_file_cluster[0][:9]:
        print(file)
        if not os.path.exists(file['file_path']):
            continue
            
        # CLEAR ENVIRONMENT
        clear_env()
        mega_purge() 
        
        # LOAD ENVIRONMENT INCL. HOSPITAL ROOM
        load_environment()
        list_cam, list_lamp, list_lamp_main, list_sun = initialize_parameters()
        
        # LOAD LIGHTING
        light_dense, sun_dense = file['light'].split(', ')[0].split('_')[-1], file['light'].split(', ')[1].split('_')[-1]
        load_lights(list_lamp, list_lamp_main[light_dense], list_sun[sun_dense])
        
        # LOAD CHARACTERS
        file_path = file['file_path']
        load_characters(file_path, file['location'], file['rotation'])
        
        # LOAD CAMERA ANGLES
        load_viewpoints(list_cam)
        
        # EXPORT IMGS
        out_name = file['file_name']
        out_dir = {'base': '//Output_IMG2/', 'file': file_path.split('.')[0].split('/')[-2]}
        export_IMG(out_name, out_dir, file['end_frame'])      
        
    # CREATE .CSV CLASS_NAMES
    create_class_names()

