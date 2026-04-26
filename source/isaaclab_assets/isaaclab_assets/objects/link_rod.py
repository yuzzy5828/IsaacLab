from __future__ import annotations
import os
import tempfile
from dataclasses import field

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

import os

@configclass
class LinkedRodObjectCfg(ArticulationCfg):
    """
    joint_positions で各ジョイント点をワールド座標で指定するロープ状オブジェクト。
    """
    joint_positions: list = field(default_factory=lambda: [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.06),
        (0.0, 0.0, 0.12),
        (0.0, 0.0, 0.18),
    ])
    link_radius: float = 0.004
    joint_type:  str   = "revolute"

    prim_path:  str                             = "{ENV_REGEX_NS}/Object"
    spawn:      sim_utils.UsdFileCfg            = None
    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    actuators: dict = None

    def __post_init__(self):
        pts = [np.array(p, dtype=float) for p in self.joint_positions]
        assert len(pts) >= 2, "joint_positions は最低2点必要です"
        n_links = len(pts) - 1

        joint_pos_dict = {f"rod_joint_{i+1}": 0.0 for i in range(n_links - 1)}
        joint_vel_dict = {f"rod_joint_{i+1}": 0.0 for i in range(n_links - 1)}
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=self.init_state.pos,
            rot=self.init_state.rot,
            joint_pos=joint_pos_dict,
            joint_vel=joint_vel_dict,
        )

        self.spawn = sim_utils.UsdFileCfg(
            usd_path=self._get_or_create_usd(pts),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #     rigid_body_enabled=True,
            #     max_linear_velocity=1000.0,
            #     max_angular_velocity=1000.0,
            #     max_depenetration_velocity=5.0,
            #     enable_gyroscopic_forces=True,
            # ),
            rigid_props=None,
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.001, rest_offset=0.0),
            mass_props=None,
            collision_props=None,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        )

        stiffness = 0.0 if self.joint_type == "revolute" else 1e6
        damping   = 0.5 if self.joint_type == "revolute" else 1e4
        if n_links > 1:
            self.actuators = {
                "rod_joints": ImplicitActuatorCfg(
                    joint_names_expr=["rod_joint_.*"],
                    stiffness=stiffness,
                    damping=damping,
                )
            }
        else:
            self.actuators = {}

    def _get_or_create_usd(self, pts: list[np.ndarray]) -> str:
        """パラメータのハッシュでファイル名を決定し、なければ生成する。"""
        import hashlib, json
        usd_dir = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/objects/linked_rod/usd"
        os.makedirs(usd_dir, exist_ok=True)

        key = json.dumps({
            "pts": [p.tolist() for p in pts],
            "r":   self.link_radius,
            "jt":  self.joint_type,
        }, sort_keys=True)
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        usd_path = os.path.join(usd_dir, f"linked_rod_{h}.usd")

        if not os.path.exists(usd_path):
            self._generate_usd(usd_path, pts)
        return usd_path

    def _generate_usd(self, usd_path: str, pts: list[np.ndarray]) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
        try:
            from pxr import Semantics
        except ImportError:
            from pxr import SemanticsSchema as Semantics

        n_links = len(pts) - 1
        stage   = Usd.Stage.CreateNew(usd_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root_path  = "/LinkedRod"
        root_xform = UsdGeom.Xform.Define(stage, root_path)
        UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())

        link_paths = []
        for i in range(n_links):
            p_start = pts[i]
            p_end   = pts[i + 1]
            vec     = p_end - p_start
            length  = float(np.linalg.norm(vec))
            center  = (p_start + p_end) / 2.0

            link_path = f"{root_path}/rod_link_{i}"
            xform     = UsdGeom.Xform.Define(stage, link_path)
            xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
            xform.AddOrientOp().Set(_rotation_from_y_to_vec(vec))

            cyl = UsdGeom.Cylinder.Define(stage, f"{link_path}/CylinderShape")
            cyl.GetRadiusAttr().Set(self.link_radius)
            cyl.GetHeightAttr().Set(length)
            cyl.GetAxisAttr().Set("Y")
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

            UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
            mass_api.GetMassAttr().Set(0.05 / n_links)

            sem = Semantics.SemanticsAPI.Apply(cyl.GetPrim(), "Semantics")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set(f"rod_link_{i}")

            link_paths.append(link_path)

        for i in range(n_links - 1):
            joint_path = f"{root_path}/rod_joint_{i + 1}"
            p_joint    = pts[i + 1]

            if self.joint_type == "fixed":
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            else:
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.GetAxisAttr().Set("X")
                joint.GetLowerLimitAttr().Set(-90.0)
                joint.GetUpperLimitAttr().Set(90.0)

            joint.GetBody0Rel().SetTargets([Sdf.Path(link_paths[i])])
            joint.GetBody1Rel().SetTargets([Sdf.Path(link_paths[i + 1])])

            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*_local_pos(p_joint, pts[i],     pts[i + 1]).tolist()))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(*_local_pos(p_joint, pts[i + 1], pts[i + 2]).tolist()))

        stage.SetDefaultPrim(root_xform.GetPrim())
        stage.Save()


# ---------- モジュールレベルのユーティリティ関数 ----------

def _local_pos(p_world: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    """ワールド座標 p_world をリンク(p_start→p_end)のローカル座標に変換する。"""
    vec    = p_end - p_start
    center = (p_start + p_end) / 2.0
    R      = _rotation_matrix_y_to_vec(vec)
    return R.T @ (p_world - center)


def _rotation_matrix_y_to_vec(vec: np.ndarray) -> np.ndarray:
    """Y軸をvec方向に向ける3x3回転行列を返す。"""
    y   = np.array([0.0, 1.0, 0.0])
    v   = vec / np.linalg.norm(vec)
    axis = np.cross(y, v)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(y, v))
    if sin_a < 1e-6:
        return np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sin_a
    K = np.array([
        [ 0,       -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def _rotation_from_y_to_vec(vec: np.ndarray):
    """Y軸をvec方向に向けるGf.Quatfを返す。"""
    from pxr import Gf
    R    = _rotation_matrix_y_to_vec(vec)
    # 回転行列 → クォータニオン
    tr   = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))