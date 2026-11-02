#!/usr/bin/env python3
"""Freeze agx_arm_description's resolved URDF + meshes into this package.

nero_moveit_config's move_group.launch.py normally resolves robot_description
live, by running xacro against agx_arm_description (a package in the separate
nero_aloha workspace) on every launch. That's fine for real bring-up, but it
means nero_bringup's offline dev bring-up would need nero_aloha sourced just
to describe the robot -- no hardware required, but an extra workspace to keep
around.

Run this once (or whenever agx_arm_description's URDF changes upstream), on a
machine that *does* have agx_arm_description sourced (e.g. after sourcing
nero_aloha's install/setup.bash). It runs the same `xacro <model>` command
move_group.launch.py uses, copies every mesh file the result references into
meshes/, writes the URDF to urdf/, and rewrites the URDF's package:// URIs to
point at this package instead -- so nero_description becomes a self-contained
snapshot and offline dev never needs agx_arm_description installed.

    python3 scripts/freeze_description.py \
        [--model nero/urdf/nero_with_gripper_description.xacro]
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_path

PACKAGE_NAME = "nero_description"
DEFAULT_MODEL = "nero/urdf/nero_with_gripper_description.xacro"
PACKAGE_URI_RE = re.compile(r'package://([^/]+)/([^"]+)')


def freeze(model: str, package_root: Path) -> Path:
    description_pkg_path = get_package_share_path("agx_arm_description")
    model_path = description_pkg_path / "agx_arm_urdf" / model

    urdf_xml = subprocess.run(
        ["xacro", str(model_path)], check=True, capture_output=True, text=True
    ).stdout

    meshes_dir = package_root / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    def rewrite(match: re.Match) -> str:
        pkg_name, rel_path = match.group(1), match.group(2)
        src = get_package_share_path(pkg_name) / rel_path
        # Namespaced by source package, in case a future model pulls meshes
        # from more than one upstream package.
        dest_rel = Path(pkg_name) / rel_path
        dest = meshes_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return f"package://{PACKAGE_NAME}/meshes/{dest_rel.as_posix()}"

    frozen_urdf = PACKAGE_URI_RE.sub(rewrite, urdf_xml)

    urdf_dir = package_root / "urdf"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = urdf_dir / Path(model).with_suffix(".urdf").name
    urdf_path.write_text(frozen_urdf)
    return urdf_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model path relative to agx_arm_description/agx_arm_urdf/, "
        "same default move_group.launch.py uses.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="This package's source root; urdf/ and meshes/ are written under it.",
    )
    args = parser.parse_args()
    urdf_path = freeze(args.model, args.package_root)
    print(f"Wrote {urdf_path} and its meshes under {args.package_root / 'meshes'}")


if __name__ == "__main__":
    main()
