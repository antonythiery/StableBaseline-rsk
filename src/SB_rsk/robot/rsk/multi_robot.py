"""
Spawn 4 instances of robot.xml in a MuJoCo scene with a floor,
run a passive viewer stepping physics in real time.
"""

import os
import copy
import time
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer

ROBOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML_PATH = os.path.join(ROBOT_DIR, "robot.xml")

# Initial (x, y, z) positions for the 4 robots
POSITIONS = [
    (-0.5, -0.5, 0.05),
    ( 0.5, -0.5, 0.05),
    (-0.5,  0.5, 0.05),
    ( 0.5,  0.5, 0.05),
]


def prefix_names(element, prefix):
    """Recursively prefix 'name' and kinematic reference attributes, leaving asset refs alone."""
    if "name" in element.attrib:
        element.attrib["name"] = prefix + element.attrib["name"]
    # Reference attributes that point to kinematic elements (not assets like mesh/material)
    for attr in ("joint", "joint1", "joint2",
                 "body", "body1", "body2",
                 "site", "site1", "site2",
                 "cranksite", "slidersite", "tendon"):
        if attr in element.attrib:
            element.attrib[attr] = prefix + element.attrib[attr]
    for child in element:
        prefix_names(child, prefix)


def build_xml():
    tree = ET.parse(ROBOT_XML_PATH)
    root = tree.getroot()

    worldbody_src = root.find("worldbody")
    actuator_src  = root.find("actuator")
    contact_src   = root.find("contact")
    asset_src     = root.find("asset")
    default_src   = root.find("default")
    compiler_src  = root.find("compiler")

    combined = ET.Element("mujoco", attrib={"model": "multi_robot"})

    # Compiler — fix meshdir to absolute path so from_xml_string can find meshes
    if compiler_src is not None:
        comp = copy.deepcopy(compiler_src)
        meshdir = comp.get("meshdir", "")
        if meshdir and not os.path.isabs(meshdir):
            comp.set("meshdir", os.path.join(ROBOT_DIR, meshdir))
        combined.append(comp)

    # Default class definitions (shared across all robots)
    if default_src is not None:
        combined.append(copy.deepcopy(default_src))

    # Assets: robot meshes/materials (shared) + floor texture/material
    combined_asset = ET.SubElement(combined, "asset")
    if asset_src is not None:
        for child in asset_src:
            combined_asset.append(copy.deepcopy(child))
    ET.SubElement(combined_asset, "texture", attrib={
        "type": "2d", "name": "groundplane", "builtin": "checker",
        "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300",
    })
    ET.SubElement(combined_asset, "material", attrib={
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2",
    })

    # Worldbody: floor + light + 4 robots
    combined_wb = ET.SubElement(combined, "worldbody")
    ET.SubElement(combined_wb, "light", attrib={
        "pos": "0 0 5", "dir": "0 0 -1", "directional": "true",
    })
    ET.SubElement(combined_wb, "geom", attrib={
        "name": "floor", "size": "0 0 0.05", "pos": "0 0 0",
        "type": "plane", "material": "groundplane",
    })

    combined_act     = ET.SubElement(combined, "actuator")
    combined_contact = ET.SubElement(combined, "contact")

    for i, (x, y, z) in enumerate(POSITIONS):
        pfx = f"r{i}_"

        # --- worldbody ---
        wb_copy = copy.deepcopy(worldbody_src)
        prefix_names(wb_copy, pfx)
        # Reposition the base body (first direct child of worldbody)
        base = wb_copy.find("body")
        if base is not None:
            base.set("pos", f"{x} {y} {z}")
        for child in wb_copy:
            combined_wb.append(child)

        # --- actuators ---
        if actuator_src is not None:
            act_copy = copy.deepcopy(actuator_src)
            prefix_names(act_copy, pfx)
            for child in act_copy:
                combined_act.append(child)

        # --- contacts ---
        # geom1/geom2 are not handled by prefix_names; floor must stay unprefixed
        if contact_src is not None:
            for pair in contact_src.findall("pair"):
                new_pair = copy.deepcopy(pair)
                for attr in ("geom1", "geom2"):
                    val = new_pair.get(attr, "")
                    if val and val != "floor":
                        new_pair.set(attr, pfx + val)
                combined_contact.append(new_pair)

    return ET.tostring(combined, encoding="unicode")


def main():
    xml_string = build_xml()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data  = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            # Sleep to maintain real-time pacing
            elapsed   = time.perf_counter() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
