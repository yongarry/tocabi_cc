#!/usr/bin/env python3
"""
stair_simulation_scene.xml 의 cube mocap body 개수를 바꿔서 재생성하는 스크립트.
사용법:  python3 gen_scene.py [개수]   (기본값 20)
"""
import sys
import os

SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "../../dyros_tocabi_v2/tocabi_description/mujoco_model/stair_simulation_scene.xml"
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 35

CUBE_TEMPLATE = '        <body name="cube_{i:02d}" mocap="true" pos="0 0 -0.05"><geom type="box" size="0.1500 0.5 0.05" rgba="0.2 0.68 0.13 1"/></body>'

cubes = "\n".join(CUBE_TEMPLATE.format(i=i) for i in range(1, N + 1))

xml = f"""<mujoco model="scene">
    <option timestep='0.0005' iterations="50" tolerance="1e-5" solver="Newton" jacobian="dense" cone="elliptic" noslip_iterations="30" noslip_tolerance="1e-5"/>
    <size njmax="8000" nconmax="4000"/>
    <compiler angle="radian" meshdir="../meshes/" balanceinertia="true"/>
    <default>
        <motor ctrllimited="true"/>
        <default class="viz">
            <geom contype="0" conaffinity="0" group="1" type="mesh" rgba=".6 .6 .7 1" />
        </default>
        <default class="cls">
            <geom group="2" rgba="0.79216 0.81961 0.93333 0.5"/>
        </default>
        <default class="cls_f">
            <geom group="2" rgba="0.79216 0.81961 0.93333 0.1" friction="1 0.005 0.0001"/>
        </default>
        <default class="FTsensor">
            <site type="cylinder" size="0.005 0.005" group="4" rgba=".1 .1 .9 1"/>
        </default>
        <default class="shg20_100_2so">
            <joint damping="0.0248" frictionloss="9.9"/>
        </default>
        <default class="shd20_100_2sh">
            <joint damping="0.0161"  frictionloss="22.0"/>
        </default>
        <default class="shg25_100_2so">
            <joint damping="0.0417"  frictionloss="14"/>
        </default>
        <default class="shg17_100_2so">
            <joint damping="0.0148"  frictionloss="6.5"/>
        </default>
        <default class="shg14_100_2so">
            <joint damping="0.0047"  frictionloss="3.7"/>
        </default>
        <default class="csf_11_100_2xh_f">
            <joint damping="0.0029" frictionloss="1.5"/>
        </default>
    </default>

    <worldbody>
        <!-- mocap body: m->body_pos / m->body_quat 으로 런타임 위치 제어 -->
        <geom name="startterrain" type="box" pos="0.1 0. -0.05" size="0.15 0.5 0.05" rgba="0.2 0.5 0.2 1"/>

{cubes}

    </worldbody>

    <visual>
        <quality shadowsize="2048" offsamples="16"/>
        <map stiffness="10" znear="0.05"/>
    </visual>
</mujoco>
"""

out = os.path.realpath(SCENE_XML)
with open(out, "w") as f:
    f.write(xml)

print(f"생성 완료: {out}  (cube 개수: {N})")
