"""
Blender 4.2 LTS Cycles GPU Prototype — Scientist #6: Willebrord Snellius (Snell's Law)
Runs headless on Free Google Colab (T4 GPU) or Kaggle (T4 x2 / P100 GPU):
    blender --background --python src/build_snellius_scene.py

Features:
  - True Cycles GPU Raytracing (OPTIX/CUDA + OpenImageDenoise)
  - Real physical Water (IOR 1.333) & Glass (IOR 1.52) Refraction Tank
  - Exact Snell's Law animated light beam: n1 * sin(theta1) = n2 * sin(theta2)
  - Cinema Camera with curved dolly + Depth of Field (f/2.2 bokeh)
  - Optional drop-in Mixamo / Ready Player Me / MPFB2 character loader (assets/character.glb or .fbx)
"""
import math
import os
import sys
import bpy
from mathutils import Vector, Euler

# ============================================================================
# 1. CONFIGURATION (10s fast test prototype -> change to 30 for full Short!)
# ============================================================================
FPS = 30
DURATION_SEC = int(os.environ.get("SNELLIUS_DURATION_SEC", "10"))
TOTAL_FRAMES = FPS * DURATION_SEC
RES_X = 1080
RES_Y = 1920
SAMPLES = int(os.environ.get("SNELLIUS_SAMPLES", "16"))
OUT_DIR = os.path.abspath(os.environ.get("SNELLIUS_FRAMES_DIR", "render_frames"))
ASSETS_DIR = os.path.abspath("assets")

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# 2. RESET SCENE & CONFIGURE CYCLES GPU (OPTIX -> CUDA -> CPU FALLBACK)
# ============================================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.render.use_persistent_data = True  # Keep static BVH/shaders in GPU VRAM across frames!
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES

scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.05
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.max_bounces = 6
scene.cycles.transmission_bounces = 6
scene.cycles.glossy_bounces = 3
scene.cycles.diffuse_bounces = 2
scene.cycles.transparent_max_bounces = 4

# Filmic / AgX cinema color management
try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - High Contrast"
except Exception:
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "High Contrast"
scene.view_settings.exposure = 0.25

# Enable GPU compute automatically if NVIDIA GPU is present
gpu_enabled = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if gpu_devices:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                if backend == "OPTIX":
                    scene.cycles.denoiser = "OPTIX"  # 10x faster NVIDIA T4 Tensor-Core denoising!
                gpu_enabled = True
                print(f"[Cycles] Enabled GPU backend: {backend} ({len(gpu_devices)} device(s)) | Denoiser: {scene.cycles.denoiser}")
                break
        except Exception:
            continue
except Exception as e:
    print(f"[Cycles] GPU setup warning: {e}")

if not gpu_enabled:
    scene.cycles.device = "CPU"
    print("[Cycles] No GPU detected -> falling back to CPU")

scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.filepath = os.path.join(OUT_DIR, "frame_")


# ============================================================================
# 3. PROCEDURAL PBR MATERIALS (TRUE CYCLES SHADERS)
# ============================================================================
def make_oak_material():
    mat = bpy.data.materials.new("LeidenOak")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.5, 8.0, 1.5)

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.5
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Distortion"].default_value = 1.4

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.045, 0.022, 0.011, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.145, 0.075, 0.036, 1.0)

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14

    bsdf.inputs["Roughness"].default_value = 0.42
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.25

    nt.links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_glass_material(name="FlintGlass", ior=1.52, tint=(0.94, 0.98, 1.0, 1.0), rough=0.02):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = tint
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["IOR"].default_value = ior
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.98
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 0.98
    return mat


def make_brass_material():
    mat = bpy.data.materials.new("AgedBrass")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.82, 0.58, 0.22, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.24
    return mat


def make_emissive_material(name, color_rgb, strength=18.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color_rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_stone_wall_material():
    mat = bpy.data.materials.new("LeidenStone")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = (0.06, 0.075, 0.11, 1.0)
    brick.inputs["Color2"].default_value = (0.04, 0.05, 0.08, 1.0)
    brick.inputs["Mortar"].default_value = (0.02, 0.025, 0.04, 1.0)
    brick.inputs["Scale"].default_value = 3.2
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bsdf.inputs["Roughness"].default_value = 0.78
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(brick.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


oak_mat = make_oak_material()
glass_mat = make_glass_material("TankGlass", ior=1.52, tint=(0.95, 0.99, 1.0, 1.0), rough=0.015)
water_mat = make_glass_material("RefractionWater", ior=1.333, tint=(0.72, 0.94, 1.0, 1.0), rough=0.01)
brass_mat = make_brass_material()
stone_mat = make_stone_wall_material()
beam_in_mat = make_emissive_material("IncidentBeam", (1.0, 0.84, 0.36), strength=24.0)
beam_out_mat = make_emissive_material("RefractedBeam", (0.22, 0.80, 1.0), strength=26.0)
normal_mat = make_emissive_material("NormalLine", (0.85, 0.92, 1.0), strength=3.0)


# ============================================================================
# 4. BUILD 1621 LEIDEN OPTICAL LAB STAGE
# ============================================================================
# World moody nocturnal blue-amber ambient
world = bpy.data.worlds.new("LeidenNight")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.012, 0.020, 0.042, 1.0)
bg.inputs["Strength"].default_value = 0.55

# Oak Workbench Tabletop (top surface at z = 0.0)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.06))
table = bpy.context.active_object
table.name = "OakWorkbench"
table.scale = (3.4, 1.8, 0.12)
table.data.materials.append(oak_mat)

# Stone Lab Backdrop Wall
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 2.2, 2.2), rotation=(math.pi * 0.5, 0.0, 0.0))
wall = bpy.context.active_object
wall.name = "StoneBackdrop"
wall.data.materials.append(stone_mat)

# Brass Pedestal Base under Refraction Tank
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.46, depth=0.045, location=(0.0, 0.0, 0.0225))
pedestal = bpy.context.active_object
pedestal.name = "BrassPedestal"
pedestal.data.materials.append(brass_mat)

# Crystal-Clear Glass Refraction Tank (Hollow Box via Solidify modifier)
TANK_W, TANK_D, TANK_H = 0.68, 0.34, 0.42
TANK_Z0 = 0.045
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, TANK_Z0 + TANK_H * 0.5))
tank = bpy.context.active_object
tank.name = "GlassTank"
tank.scale = (TANK_W, TANK_D, TANK_H)
bpy.ops.object.transform_apply(scale=True)
solid = tank.modifiers.new("GlassThickness", "SOLIDIFY")
solid.thickness = 0.012
tank.data.materials.append(glass_mat)

# Water Volume inside lower half of tank (Air -> Water interface at WATER_SURFACE_Z)
WATER_H = 0.25
WATER_SURFACE_Z = TANK_Z0 + WATER_H  # 0.295m
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, TANK_Z0 + WATER_H * 0.5 + 0.008))
water = bpy.context.active_object
water.name = "WaterVolume"
water.scale = (TANK_W - 0.028, TANK_D - 0.028, WATER_H - 0.014)
bpy.ops.object.transform_apply(scale=True)
water.data.materials.append(water_mat)

# Brass Protractor Ring behind tank (centered on refraction hit point!)
HIT_POINT = Vector((0.0, 0.0, WATER_SURFACE_Z))
bpy.ops.mesh.primitive_torus_add(
    major_radius=0.34,
    minor_radius=0.010,
    major_segments=64,
    minor_segments=16,
    location=(0.0, 0.19, WATER_SURFACE_Z),
    rotation=(math.pi * 0.5, 0.0, 0.0),
)
protractor = bpy.context.active_object
protractor.name = "BrassProtractor"
protractor.data.materials.append(brass_mat)

# Vertical Normal Reference Line (dashed/thin glowing cylinder through HIT_POINT)
bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.0035, depth=0.64, location=(0.0, 0.0, WATER_SURFACE_Z))
normal_line = bpy.context.active_object
normal_line.name = "SurfaceNormal"
normal_line.data.materials.append(normal_mat)

# Glowing Point-of-Incidence Spark Orb
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.018, location=HIT_POINT)
hit_orb = bpy.context.active_object
hit_orb.name = "RefractionNode"
hit_orb.data.materials.append(beam_in_mat)

# 3D Golden Formula Floating Above Tank: n1 sin θ1 = n2 sin θ2
bpy.ops.object.text_add(location=(0.0, 0.05, 0.78), rotation=(math.pi * 0.5, 0.0, 0.0))
formula = bpy.context.active_object
formula.name = "SnellFormula"
formula.data.body = "n1 sin θ1 = n2 sin θ2"
formula.data.align_x = "CENTER"
formula.data.size = 0.095
formula.data.extrude = 0.012
formula.data.bevel_depth = 0.003
formula.data.materials.append(brass_mat)


# ============================================================================
# 5. ANIMATED SNELL'S LAW LIGHT BEAMS (EXACT PHYSICS PER FRAME!)
# ============================================================================
def create_beam_cylinder(name, mat, radius=0.011):
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


beam_in = create_beam_cylinder("IncidentLightBeam", beam_in_mat, radius=0.010)
beam_out = create_beam_cylinder("RefractedLightBeam", beam_out_mat, radius=0.010)

# Brass Lantern Emitter riding the top of the incident beam
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.032, depth=0.09, location=(0, 0, 0))
lantern = bpy.context.active_object
lantern.name = "BrassLightEmitter"
lantern.data.materials.append(brass_mat)


def orient_cylinder_between(obj, p1, p2):
    diff = p2 - p1
    length = diff.length
    obj.location = (p1 + p2) * 0.5
    obj.scale = (1.0, 1.0, max(0.001, length))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = diff.normalized().to_track_quat("Z", "Y")


N1 = 1.0003  # Air
N2 = 1.3330  # Water
IN_LEN = 0.52
OUT_LEN = 0.25

for f in range(1, TOTAL_FRAMES + 1):
    u = (f - 1) / max(1, TOTAL_FRAMES - 1)
    # Smoothly sweep incidence angle theta1 from 16 deg -> 64 deg -> 38 deg
    theta1_deg = 16.0 + 48.0 * math.sin(u * math.pi * 0.85)
    theta1 = math.radians(theta1_deg)
    # Exact Snell's Law: sin(theta2) = (n1 / n2) * sin(theta1)
    sin_t2 = max(-0.999, min(0.999, (N1 / N2) * math.sin(theta1)))
    theta2 = math.asin(sin_t2)

    p_start = HIT_POINT + Vector((-math.sin(theta1) * IN_LEN, 0.0, math.cos(theta1) * IN_LEN))
    p_end = HIT_POINT + Vector((math.sin(theta2) * OUT_LEN, 0.0, -math.cos(theta2) * OUT_LEN))

    orient_cylinder_between(beam_in, p_start, HIT_POINT)
    orient_cylinder_between(beam_out, HIT_POINT, p_end)

    lantern.location = p_start
    lantern.rotation_mode = "QUATERNION"
    lantern.rotation_quaternion = (HIT_POINT - p_start).normalized().to_track_quat("Z", "Y")

    for ob in (beam_in, beam_out, lantern):
        ob.keyframe_insert(data_path="location", frame=f)
        ob.keyframe_insert(data_path="rotation_quaternion", frame=f)
        if ob != lantern:
            ob.keyframe_insert(data_path="scale", frame=f)


# ============================================================================
# 6. OPTIONAL DROP-IN REALISTIC CHARACTER LOADER (MIXAMO / MPFB2 / .GLB / .FBX)
# ============================================================================
for fname in ("character.glb", "character.gltf", "character.fbx"):
    fpath = os.path.join(ASSETS_DIR, fname)
    if os.path.isfile(fpath):
        print(f"[Assets] Auto-importing realistic character: {fpath}")
        before = set(bpy.data.objects)
        if fname.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=fpath)
        else:
            bpy.ops.import_scene.gltf(filepath=fpath)
        imported = [o for o in bpy.data.objects if o not in before]
        # Place character standing naturally behind-left of the optical workbench
        for o in imported:
            if o.parent is None:
                o.location = Vector((-0.62, 0.55, -0.82))
                o.rotation_euler = Euler((0.0, 0.0, math.radians(-22.0)))
        break


# ============================================================================
# 7. CINEMA LIGHTING (3-POINT CHIAROSCURO) & CURVED DOLLY CAMERA WITH DOF
# ============================================================================
# Warm Candlelight Key Area Light
bpy.ops.object.light_add(type="AREA", location=(-1.4, -1.2, 1.35))
key_light = bpy.context.active_object
key_light.data.energy = 145.0
key_light.data.size = 0.85
key_light.data.color = (1.0, 0.78, 0.52)
key_light.rotation_euler = Euler((math.radians(52), 0.0, math.radians(-42)))

# Cool Sapphire Moonlight Rim Light
bpy.ops.object.light_add(type="AREA", location=(1.5, 1.4, 1.55))
rim_light = bpy.context.active_object
rim_light.data.energy = 180.0
rim_light.data.size = 1.1
rim_light.data.color = (0.48, 0.76, 1.0)
rim_light.rotation_euler = Euler((math.radians(58), 0.0, math.radians(138)))

# Subtle Cyan Under-Tank Bounce
bpy.ops.object.light_add(type="POINT", location=(0.0, -0.35, 0.25))
bounce = bpy.context.active_object
bounce.data.energy = 14.0
bounce.data.color = (0.35, 0.82, 1.0)

# Camera Target Empty locked to Refraction Interface
bpy.ops.object.empty_add(location=(0.0, 0.0, 0.34))
cam_target = bpy.context.active_object
cam_target.name = "CameraTarget"

bpy.ops.object.camera_add(location=(0.0, -1.75, 0.58))
cam = bpy.context.active_object
cam.name = "CinemaCamera"
scene.camera = cam
cam.data.lens = 50.0
cam.data.dof.use_dof = True
cam.data.dof.focus_object = hit_orb
cam.data.dof.aperture_fstop = 2.4

track = cam.constraints.new(type="TRACK_TO")
track.target = cam_target
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"

# Smooth curved cinematic orbital push-in across 10 seconds
for f in (1, TOTAL_FRAMES // 2, TOTAL_FRAMES):
    u = (f - 1) / max(1, TOTAL_FRAMES - 1)
    ang = math.radians(-18.0 + 36.0 * u)
    radius = 1.65 - 0.32 * math.sin(u * math.pi)
    cam.location = Vector((math.sin(ang) * radius, -math.cos(ang) * radius, 0.54 + 0.14 * math.sin(u * math.pi)))
    cam.keyframe_insert(data_path="location", frame=f)

print(f"[Snellius Prototype] Scene ready! Rendering {TOTAL_FRAMES} frames ({RES_X}x{RES_Y}, {SAMPLES} samples) to {OUT_DIR} ...")
bpy.ops.render.render(animation=True)
print("[Snellius Prototype] Render complete!")
